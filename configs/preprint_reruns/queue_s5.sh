#!/bin/bash
cd /home/luke/ATOM
B=/home/luke/ATOM/configs/preprint_reruns
while ! grep -q "ALL DONE" $B/logs/status_tau375.log; do sleep 60; done
for d in s5_md17_noneq_rest s5_toluene_tau3000; do
  echo "START $d $(date)" >> $B/logs/status_tau375.log
  .venv/bin/train --configs $B/tau375/$d > $B/logs/tau375_$d.log 2>&1
  echo "END $d exit=$? $(date)" >> $B/logs/status_tau375.log
done
echo "S5 DONE $(date)" >> $B/logs/status_tau375.log
