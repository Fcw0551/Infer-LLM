"""Run the four KV-cache profiles serially and record Linux resource pressure.
No global VM settings or other processes are changed. Raw monitoring includes warmup.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]

def snapshot(pid=None):
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    cpu = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    pressure = {}
    for kind in ('cpu', 'memory', 'io'):
        for line in Path('/proc/pressure/' + kind).read_text().splitlines():
            fields = line.split()
            pressure[kind + '_' + fields[0]] = int(dict(x.split('=') for x in fields[1:])['total'])
    ticks = 0
    if pid:
        try:
            fields = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()
            ticks = int(fields[11]) + int(fields[12])
        except FileNotFoundError:
            pass
    result = dict(monotonic=time.monotonic(), mem_available_mib=mem['MemAvailable']/1024,
                swap_used_mib=(mem['SwapTotal']-mem['SwapFree'])/1024,
                pswpin=int(vm['pswpin']), pswpout=int(vm['pswpout']),
                cpu_total=sum(cpu), cpu_busy=sum(cpu)-cpu[3]-cpu[4],
                cpu_iowait=cpu[4], process_ticks=ticks, **pressure)
    fields = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    for key, value in zip(('gpu_util_percent','gpu_memory_used_mib','gpu_temperature_c','gpu_power_w'), fields):
        try:
            result[key] = float(value.strip())
        except ValueError:
            result[key] = ''
    return result

def idle_gate():
    for attempt in range(20):
        a = snapshot()
        time.sleep(3)
        b = snapshot()
        busy = (b['cpu_busy']-a['cpu_busy'])/max(1, b['cpu_total']-a['cpu_total'])
        if (busy < .10 and b['mem_available_mib'] > 4096 and
            isinstance(b['gpu_util_percent'], (int,float)) and b['gpu_util_percent'] < 10 and
            b['pswpin'] == a['pswpin'] and b['pswpout'] == a['pswpout'] and
            b['memory_full'] == a['memory_full']):
            return b
    raise RuntimeError('系统未达到空闲/内存门槛，停止测试以免混入压力数据')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='docs/performance/cuda-contiguous-kv-20261107')
    args = parser.parse_args()
    dest = ROOT / args.output
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob('p[0-9]*.csv')):
        raise RuntimeError('输出目录已有测量数据，请使用新目录以免覆盖')
    def command_text(command):
        return subprocess.check_output(command, text=True, cwd=ROOT).strip()
    model = ROOT/'models/Qwen3-0.6B-F32.gguf'
    digest = hashlib.sha256()
    with model.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(chunk)
    manifest = dict(git=command_text(['git','rev-parse','HEAD']),
        git_status=command_text(['git','status','--short']),
        gpu=command_text(['nvidia-smi','--query-gpu=name,driver_version,compute_cap,memory.total','--format=csv,noheader']),
        nvcc=command_text(['/usr/local/cuda/bin/nvcc','--version']),
        gcc=command_text(['/usr/bin/g++-13','--version']),
        kernel=command_text(['uname','-r']), os_release=Path('/etc/os-release').read_text(),
        model_bytes=model.stat().st_size, model_sha256=digest.hexdigest(),
        backend='cuda:0 + CPU embedding', build='Release', architecture='89',
        workers=8, warmup=1, repetitions=3, output_tokens=128,
        gpu_sampling='whole GPU, 1 second intervals, includes warmup; not process allocation peak')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files = sorted(list((ROOT/'src').rglob('*.cpp')) + list((ROOT/'src').rglob('*.cu')) + list((ROOT/'include').rglob('*.hpp')) + [Path(__file__), ROOT/'benchmarks/cuda/bench.cpp', ROOT/'benchmarks/cuda/CMakeLists.txt', ROOT/'CMakeLists.txt'])
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (dest/'source-hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    summaries = {}
    for p in (32,64,128,256):
        print('Waiting for idle system: p%d' % p, flush=True)
        before = idle_gate()
        command = [str(ROOT/'benchmarks/cuda/build/inferllm-bench'), '--profile', 'p%d'%p,
                   '-t','8','-r','3','--warmup','1','--csv',str(dest/('p%d.csv'%p))]
        print('Running p%d: 1 warmup + 3 measured requests, 128 outputs each' % p, flush=True)
        samples = [before]
        with (dest/('p%d.log'%p)).open('w') as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            with (dest/('p%d-system.csv'%p)).open('w') as monitor:
                writer = csv.DictWriter(monitor, fieldnames=list(before))
                writer.writeheader()
                writer.writerow(before)
                while proc.poll() is None:
                    sample = snapshot(proc.pid)
                    samples.append(sample)
                    writer.writerow(sample)
                    monitor.flush()
                    if sample['mem_available_mib'] < 512:
                        proc.terminate()
                        proc.wait()
                        raise RuntimeError('剩余内存低于 512 MiB，终止本次 benchmark')
                    time.sleep(1)
                code = proc.wait()
                after = snapshot()
                writer.writerow(after)
                samples.append(after)
        intervals = []
        for a,b in zip(samples, samples[1:]):
            if a['process_ticks'] and b['process_ticks']:
                total = b['cpu_total']-a['cpu_total']
                external = max(0, b['cpu_busy']-a['cpu_busy']-(b['process_ticks']-a['process_ticks']))
                intervals.append((external,total))
        summary = dict(exit_code=code, seconds=after['monotonic']-before['monotonic'],
            min_mem_available_mib=min(s['mem_available_mib'] for s in samples),
            max_swap_used_mib=max(s['swap_used_mib'] for s in samples),
            swapin_pages=after['pswpin']-before['pswpin'], swapout_pages=after['pswpout']-before['pswpout'],
            memory_full_stall_ms=(after['memory_full']-before['memory_full'])/1000,
            memory_some_stall_ms=(after['memory_some']-before['memory_some'])/1000,
            io_full_stall_ms=(after['io_full']-before['io_full'])/1000,
            external_cpu_percent=100*sum(x for x,y in intervals)/max(1,sum(y for x,y in intervals)))
        for key in ('gpu_util_percent','gpu_memory_used_mib','gpu_temperature_c','gpu_power_w'):
            values = [s[key] for s in samples if isinstance(s[key], (int,float))]
            summary['max_'+key] = max(values) if values else None
        summary['gpu_memory_before_mib'] = before['gpu_memory_used_mib']
        summary['pressure_flag'] = (summary['swapin_pages'] > 0 or summary['swapout_pages'] > 0 or
            summary['memory_full_stall_ms'] > 0 or summary['min_mem_available_mib'] < 1024 or
            summary['external_cpu_percent'] > 5)
        summaries['p%d'%p] = summary
        (dest/'system-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print('Finished p%d: %s' % (p,json.dumps(summary)), flush=True)
        if code or summary['pressure_flag']:
            raise RuntimeError('运行失败或观测到资源干扰，保留原始数据，停止后续测试')
    changed = [str(p.relative_to(ROOT)) for p in files if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]]
    if changed:
        raise RuntimeError('测试期间源码变化：'+str(changed))

if __name__ == '__main__':
    main()
