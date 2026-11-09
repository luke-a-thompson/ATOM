#!/bin/bash
cd /home/luke/ATOM
B=/home/luke/ATOM/configs/preprint_reruns
for d in s6_md17_last s6_md22_gat s6_tg80_random; do
  echo "START $d $(date)" >> $B/logs/status_tau375.log
  .venv/bin/train --configs $B/tau375/$d > $B/logs/tau375_$d.log 2>&1
  echo "END $d exit=$? $(date)" >> $B/logs/status_tau375.log
done
echo "S6 DONE $(date)" >> $B/logs/status_tau375.log
