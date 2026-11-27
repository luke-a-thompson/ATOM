B=/home/luke/ATOM/configs/preprint_reruns
cd /home/luke/ATOM
for d in s11_tg80_egno_fix s11_tg80_egno_id_fix s11_tg80_egnn_r_fix s11_tg80_egnn_s_fix; do
  echo "START $d $(date)" >> $B/logs/status_tau375.log
  .venv/bin/train --configs $B/tau375/$d > $B/logs/tau375_$d.log 2>&1
  echo "END $d exit=$? $(date)" >> $B/logs/status_tau375.log
done
echo "S11 DONE $(date)" >> $B/logs/status_tau375.log
