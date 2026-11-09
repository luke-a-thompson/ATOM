#!/bin/bash
cd /home/luke/ATOM
B=/home/luke/ATOM/configs/preprint_reruns
while ! grep -q "S6 DONE" $B/logs/status_tau375.log; do sleep 60; done
for d in s7_md17_egno s7_md17_last_egno; do
  echo "START $d $(date)" >> $B/logs/status_tau375.log
  .venv/bin/train --configs $B/tau375/$d > $B/logs/tau375_$d.log 2>&1
  echo "END $d exit=$? $(date)" >> $B/logs/status_tau375.log
done
echo "S7 DONE $(date)" >> $B/logs/status_tau375.log
