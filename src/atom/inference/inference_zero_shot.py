import subprocess

subprocess.run(
    [
        "poetry",
        "run",
        "inference",
        "--model",
        "benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold1_multitask_15-May-2025_09-36-35/run_1/best_val_model.pth",
        "--config",
        "configs/tg80_multitask/atom_multitask_muon_fold1.toml",
    ],
    shell=True,
)
