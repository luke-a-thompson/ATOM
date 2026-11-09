#!/bin/bash
cd /home/luke/ATOM
while pgrep -f "train --configs configs/preprint_reruns/probe_tau" >/dev/null; do sleep 30; done
FIX_INIT=1 FIX_SWIGLU=1 .venv/bin/python -W ignore configs/preprint_reruns/probe_fixes.py --configs configs/preprint_reruns/probe_fix/both > configs/preprint_reruns/logs/probe_fix.log 2>&1
echo "PROBES DONE $(date)" >> configs/preprint_reruns/logs/status.log
