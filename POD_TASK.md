# Pod task: stochastic-ΔT retrain (ATOM + NoPE) and ΔT sweep

Goal: fix the ΔT-invariance figure in the paper. Currently ATOM was trained at
fixed ΔT=3000 fs, so it only does well at ΔT=3000. We need ATOM and NoPE models
trained with stochastic ΔT. EGNO is NOT retrained.

0. Setup: install uv if missing, then `uv sync`. Read CLAUDE.md.
1. Data: download TG80 from https://ses.library.usyd.edu.au/handle/2123/35120
   and convert it to data/tg80_npz/ using data/TG80.py. Only Cluster 1
   (in-distribution split, fold 1) molecules are needed.
2. Configs: copy configs/preprint_reruns/tg80_dt_sweep/dt_atom.toml and
   dt_nope.toml into configs/preprint_reruns/tg80_dt_stochastic/ and set
   delta_T = [10, 10000] (stochastic ΔT; the code supports a [min, max] range).
   Keep everything else the same.
3. Train ATOM then NoPE on Cluster 1, one after another, in the background with
   nohup, logging to files. Check that the first epochs run before leaving it.
4. Run the ΔT sweep (P=8, ΔT = 10, 30, 100, 300, 1000, 3000, 10000) on the two
   new models, the same way results/preprint/invariance_results/t_sweep_tau375.json
   was produced (see results/preprint/README.md and
   Z_paper_content/invariance_sweep.py). For EGNO, reuse the EGNO series from
   t_sweep_tau375.json unchanged. Save the combined result to
   results/preprint/invariance_results/t_sweep_stochastic.json.
5. Regenerate paper/Figures/invariances/t_invariance_multi.pdf from it.
6. Commit only the configs, the results JSON (+ each run's results.json and
   config, no checkpoints) and the figure. Push to claude/happy-curie-8y5do3.
7. Report a short table: mean S2T MSE per model per ΔT, and whether ATOM now
   beats EGNO and NoPE across ΔT.

Don't edit the paper text and don't push to Overleaf.
