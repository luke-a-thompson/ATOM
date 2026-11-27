B=/home/luke/ATOM/configs/preprint_reruns
cd /home/luke/ATOM
echo "START s10_tg80_nope_id $(date)" >> $B/logs/status_tau375.log
.venv/bin/train --configs $B/tau375/s10_tg80_nope_id > $B/logs/tau375_s10.log 2>&1
echo "END s10_tg80_nope_id exit=$? $(date)" >> $B/logs/status_tau375.log
echo "S10 DONE $(date)" >> $B/logs/status_tau375.log
