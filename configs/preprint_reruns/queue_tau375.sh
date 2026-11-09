#!/bin/bash
cd /home/luke/ATOM
B=/home/luke/ATOM/configs/preprint_reruns
for d in s1_md17_atom s1_rmd17_atom s1_md22_atom s1_md17_noneq s2_tg80_atom s2_tg80_atom_id s3_equivariance_error s4_tg80_dt_sweep s4_tg80_formic s4_tg80_ablation s4_md17_ablation; do
  echo "START $d $(date)" >> $B/logs/status_tau375.log
  .venv/bin/train --configs $B/tau375/$d > $B/logs/tau375_$d.log 2>&1
  echo "END $d exit=$? $(date)" >> $B/logs/status_tau375.log
done
echo "ALL DONE $(date)" >> $B/logs/status_tau375.log
