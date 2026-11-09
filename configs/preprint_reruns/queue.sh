#!/bin/bash
cd /home/luke/ATOM
for d in tg80_dt_sweep tg80_formic tg80_ablation md17_ablation; do
  echo "START $d $(date)" >> /home/luke/ATOM/configs/preprint_reruns/logs/status.log
  .venv/bin/train --configs /home/luke/ATOM/configs/preprint_reruns/$d > /home/luke/ATOM/configs/preprint_reruns/logs/$d.log 2>&1
  echo "END $d exit=$? $(date)" >> /home/luke/ATOM/configs/preprint_reruns/logs/status.log
done
echo "ALL DONE $(date)" >> /home/luke/ATOM/configs/preprint_reruns/logs/status.log
