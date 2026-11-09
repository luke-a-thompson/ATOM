B=/home/luke/ATOM/configs/preprint_reruns
cd /home/luke/ATOM
until grep -q "S7 DONE" $B/logs/status_tau375.log; do sleep 60; done
mv benchmark_runs/s6_md17_last_09-Nov-2026_19-04-14 benchmark_runs/INVALID_s6_md17_last_uniform_bug
echo "START s9_md17_last_rerun $(date)" >> $B/logs/status_tau375.log
.venv/bin/train --configs $B/tau375/s6_md17_last > $B/logs/tau375_s9_md17_last.log 2>&1
echo "END s9_md17_last_rerun exit=$? $(date)" >> $B/logs/status_tau375.log
echo "S9 DONE $(date)" >> $B/logs/status_tau375.log
