# Preprint results

`results.json` + training config for every run used in the preprint (T-RoPE τ=375 unless noted). Checkpoints and logs are not included.

| Paper item | Run dir |
|---|---|
| MD17 Table 1 (ATOM) | s1_md17_atom |
| MD17 Table 1 (EGNO) | s7_md17_egno |
| rMD17 table | s1_rmd17_atom, rmd17_egno |
| MD22 table | s1_md22_atom, s6_md22_gat (ATOM-GATv2) |
| TG80 OOD / ID | s2_tg80_atom, s2_tg80_atom_id, tg80_egno, tg80_egno_id, tg80_egnn_r, tg80_egnn_s |
| TG80 random split | s6_tg80_random |
| Formic acid | s4_tg80_formic |
| Equivariance defect | s3_equivariance_error, s1_md17_noneq, s5_md17_noneq_rest |
| Ablations | s4_md17_ablation, s4_tg80_ablation |
| MD17 final-frame | s6_md17_last (ATOM), s7_md17_last_egno |
| ΔT sweep (fixed-ΔT training) | s4_tg80_dt_sweep, s8_dt_tau24000 (τ=24000) |
| P / ΔT sweep outputs | invariance_results/ |
