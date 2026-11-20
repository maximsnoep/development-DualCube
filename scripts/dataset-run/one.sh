#!/bin/bash
# One mesh (argument) through every stage with a time limit (LIMIT seconds, default 1200); kills the whole
# process group (DiskScissors too). Records why a run was killed: timeout, out of memory, or other.
R=${RUN:?set RUN to the run directory}
mesh=$1
limit=${LIMIT:-1200}
folder=$(echo "$mesh" | sed 's|.*/dataset/||; s|/.*||')
name=$(echo "$mesh" | sed 's|.*/dataset/||; s|/|__|g; s|\.obj$||')
mkdir -p "$R/rows" "$R/logs"
rm -f "$R/logs/$name.timeout"
start=$(date +%s)
setsid "$R/cli" stages --input "$mesh" --report "$R/rows/$name.csv" > "$R/logs/$name.out" 2> "$R/logs/$name.err.full" &
pid=$!
( sleep "$limit"; touch "$R/logs/$name.timeout"; kill -KILL -- -$pid 2>/dev/null ) & watcher=$!
wait $pid; code=$?
kill $watcher 2>/dev/null; kill -KILL -- -$pid 2>/dev/null
# Why it ended: the watchdog leaves a marker; the kernel logs OOM kills by pid (cli or a child).
killed=none
if [ -e "$R/logs/$name.timeout" ]; then
  killed=timeout
elif [ $code -eq 137 ]; then
  if dmesg 2>/dev/null | grep -qE "Out of memory: Killed process $pid \(|oom-kill:.*pid=$pid,"; then
    killed=oom
  else
    killed=other
  fi
fi
rm -rf /tmp/dualcube-certify-$pid-* /tmp/dualcube-tetgen-$pid-*
grep -a "TetGen added" "$R/logs/$name.err.full" > "$R/logs/$name.tetgen"
grep -a "Stage .* started" "$R/logs/$name.err.full" | tail -n 1 > "$R/logs/$name.stage"
tail -n 40 "$R/logs/$name.err.full" > "$R/logs/$name.err"; rm -f "$R/logs/$name.err.full"
echo "$name folder=$folder exit=$code seconds=$(( $(date +%s) - start )) killed=$killed" >> "$R/done.txt"
