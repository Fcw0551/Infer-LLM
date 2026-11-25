"""Replay recorded assistant turns without executing recorded commands."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
for name in ("worker", "model", "trajectory", "report", "output"):
    parser.add_argument(name, type=Path)
parser.add_argument("--block-size", type=int, default=16)
parser.add_argument("--budget", type=int, default=1024)
parser.add_argument("--context", type=int, default=16384)
args = parser.parse_args()
raw = args.trajectory.read_bytes()
trajectory = json.loads(raw)
instance = trajectory["instance_id"]
report = json.loads(args.report.read_text())
if report[instance]["resolved"] is not True:
    raise ValueError("expected a resolved trajectory")

args.output.mkdir()
with (args.output / "rounds.jsonl").open("w", buffering=1) as log, (
    args.output / "worker.stderr"
).open("wb") as errors, subprocess.Popen(
    [args.worker, args.model, args.output / "blocks.jsonl", str(args.block_size), str(args.budget), str(args.context)],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
) as worker:
    log.write(json.dumps({"instance": instance, "sha256": hashlib.sha256(raw).hexdigest(),
                          "block_size": args.block_size, "budget_per_head": args.budget, "context": args.context}) + "\n")
    pending = []
    call = 0
    for message in trajectory["messages"]:
        role = message["role"]
        content = (message.get("content") or "").strip()
        if role != "assistant":
            pending.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            continue
        call += 1
        for item in message.get("tool_calls") or []:
            content += "\n<tool_call>\n" + json.dumps(item["function"], ensure_ascii=False) + "\n</tool_call>"
        body = ("".join(pending) + "<|im_start|>assistant\n").encode()
        reply = (content + "<|im_end|>\n").encode()
        worker.stdin.write(f"{len(body)} {len(reply)}\n".encode() + body + reply)
        worker.stdin.flush()
        response = worker.stdout.readline().decode().split()
        if not response:
            raise RuntimeError("replay worker exited; see worker.stderr")
        status, start, input_tokens, reply_tokens = response
        log.write(json.dumps(dict(call=call, status=status, start=int(start),
                                  input_tokens=int(input_tokens), reply_tokens=int(reply_tokens))) + "\n")
        if status != "ok":
            break
        pending.clear()
    worker.stdin.close()
    if worker.wait():
        raise RuntimeError("replay worker failed")
