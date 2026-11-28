B=/home/luke/ATOM/configs/preprint_reruns
cd /home/luke/ATOM
d=s12_tg80_random_egno
echo "START $d $(date)" >> $B/logs/status_tau375.log
.venv/bin/train --configs $B/tau375/$d > $B/logs/tau375_$d.log 2>&1
echo "END $d exit=$? $(date)" >> $B/logs/status_tau375.log
echo "S12 DONE $(date)" >> $B/logs/status_tau375.log
