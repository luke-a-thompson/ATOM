# ATOM: A Pretrained Neural Operator for Multitask Dynamics Learning

This repository is the official implementation of [ATOM: A Pretrained Neural Operator for Multitask Dynamics Learning](PLACEHOLDER). ATOM is a graph transformer neural operator for the parallel decoding of molecular dynamics trajectory. We show state-of-the-art performance on the existing MD17 dataset, and for the first time, demonstrate zero-shot generalization to unseen chemical compounds.

![ATOM Diagram](Z_paper_content/readme_content/arch.png)

## Command overview

### Training and inference

- **Train a single model**

```bash
uv run train --config path/to/config.toml
```

- **Train many models (e.g. ablations)**

```bash
uv run train --configs path/to/folder_with_configs
```

- **Run inference with a trained model**

```bash
uv run inference --model path/to/model.pth --config path/to/config.toml
```

### Tables and analysis (paper results)

- **Generate MSE tables** (LaTeX tables under `Z_paper_content/tables`)

```bash
uv run paper tables mse path/to/egno_runs path/to/atom_runs
```

- **Generate runtime tables** (LaTeX runtime comparison under `Z_paper_content/tables`)

```bash
uv run paper tables runtime path/to/egno_runs path/to/atom_runs --dataset <dataset_name>
```

- **Generate all tables from one or more run directories** (MSE + optional runtime tables)

```bash
uv run paper tables all path/to/run_dir_1 path/to/run_dir_2
```

- **Analyze multitask folds (EGNO vs ATOM)** (per-fold stats and % improvements, printed to stdout)

```bash
uv run paper analyze folds path/to/egno_multitask_runs path/to/atom_multitask_runs
```

- **Analyze raw trajectory datasets (MD17 / RMD17 / TG80)** (drift vs internal motion scatter plots)

```bash
uv run paper analyze dataset data/md17_npz --dataset md17
uv run paper analyze dataset data/rmd17_npz --dataset rmd17
uv run paper analyze dataset data/tg80_npz --dataset tg80
```

- **Benchmark ATOM inference timing** (writes JSON summary under `Z_paper_content/` by default)

```bash
uv run paper analyze timing --atom-run-dir path/to/md17_atom_runs --num-repeats 3
```

### Figures for the paper

- **Ablation bar plots** (saved under `Z_paper_content/ablations`)

```bash
uv run paper figures ablations path/to/ablations_runs
```

- **P-invariance curves (num_timesteps)** (saved under `Z_paper_content/invariance_results` by default)

```bash
uv run paper figures invariance-p --p "[4,8,12]" --config path/to/config.toml --model path/to/checkpoints_dir --save-dir Z_paper_content/invariance_results
```

- **T-invariance curves (Δt)** (saved under `Z_paper_content/invariance_results` by default)

```bash
uv run paper figures invariance-t --t "[1,2,4,8]" --config path/to/config.toml --model path/to/checkpoints_dir --save-dir Z_paper_content/invariance_results
```

- **Multitask scaling figures** (saved to a PDF you choose, typically under `Z_paper_content/multitask_scaling`)

```bash
uv run paper figures multitask-scaling path/to/tg80_multitask_scaling_runs --output Z_paper_content/multitask_scaling/multitask_scaling.pdf
```

- **Trajectory visualisations (MD17 / RMD17 / TG80)** (saved under `Z_paper_content/trajectories`)

```bash
uv run paper figures trajectories data/tg80_npz --dataset tg80
```

## Requirements

To install requirements:

```setup
poetry install --with dev
```

The results were gathered on Cuda 12.4.

## Training

To train ATOM, run this command:

```bash
poetry run train --config <<path_to_config.toml>>
```

to train multiple models (e.g., for the purpose of ablations) run:

```bash
poetry run train --configs <<path_to_folder_containing_configs>>
```

To edit model hyperparameters, please edit the config.toml files. Feel free to experiment! A Pydantic validator will ensure your hyperparameter choices do not cause unforeseen issues :).

## Evaluation

To inference ATOM run the command:
```bash
poetry run train --model <<path_to_model.pth>> --config <<path_to_config.toml>>
```

## Evaluating Equivariance Error

<details>
<summary>Evaluating Loss Robustness to Input Rotations</summary>

### ATOM
```bash
uv run rotation_loss_robustness --config benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_aspirin_25-Sep-2025_03-36-08/md_aspirin_25-Sep-2025_03-36-08.toml --model benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_aspirin_25-Sep-2025_03-36-08/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

### ATOM No Equivariant Lift
```bash
uv run rotation_loss_robustness --config benchmark_runs/ablations_atom_17-Sep-2025_00-38-16/no_equivariant_lifting_17-Sep-2025_00-38-16/no_equivariant_lifting_17-Sep-2025_00-38-16.toml --model benchmark_runs/ablations_atom_17-Sep-2025_00-38-16/no_equivariant_lifting_17-Sep-2025_00-38-16/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

</summmary>
</details>

<details>
<summary>Evaluating Monte Carlo Quasi-equivariance Error</summary>

### ATOM
```bash
uv run equivariance_defect --config benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_aspirin_25-Sep-2025_03-36-08/md_aspirin_25-Sep-2025_03-36-08.toml --model benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_aspirin_25-Sep-2025_03-36-08/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_ethanol_25-Sep-2025_03-36-08/md_ethanol_25-Sep-2025_03-36-08.toml --model benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_ethanol_25-Sep-2025_03-36-08/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_malonaldehyde_25-Sep-2025_03-36-08/md_malonaldehyde_25-Sep-2025_03-36-08.toml --model benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_malonaldehyde_25-Sep-2025_03-36-08/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_naphtalene_25-Sep-2025_03-36-08/md_naphtalene_25-Sep-2025_03-36-08.toml --model benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_naphtalene_25-Sep-2025_03-36-08/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_salicylic_25-Sep-2025_03-36-08/md_salicylic_25-Sep-2025_03-36-08.toml --model benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_salicylic_25-Sep-2025_03-36-08/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_toluene_25-Sep-2025_03-36-08/md_toluene_25-Sep-2025_03-36-08.toml --model benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_toluene_25-Sep-2025_03-36-08/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_uracil_25-Sep-2025_03-36-08/md_uracil_25-Sep-2025_03-36-08.toml --model benchmark_runs/md17/md17_uniform_paper_atom_25-Sep-2025_03-36-08/md_uracil_25-Sep-2025_03-36-08/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

### ATOM No Equivariant Lift

```bash
uv run equivariance_defect --config benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_aspirin_19-Nov-2025_22-49-24/md_aspirin_19-Nov-2025_22-49-24.toml --model benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_ethanol_19-Nov-2025_22-49-24/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_ethanol_19-Nov-2025_22-49-24/md_ethanol_19-Nov-2025_22-49-24.toml --model benchmark_runs/ablations_atom_17-Sep-2025_00-38-16/no_equivariant_lifting_17-Sep-2025_00-38-16/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_malonaldehyde_19-Nov-2025_22-49-24/md_malonaldehyde_19-Nov-2025_22-49-24.toml --model benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_malonaldehyde_19-Nov-2025_22-49-24/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_naphtalene_19-Nov-2025_22-49-24/md_naphtalene_19-Nov-2025_22-49-24.toml --model benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_naphtalene_19-Nov-2025_22-49-24/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_salicylic_19-Nov-2025_22-49-24/md_salicylic_19-Nov-2025_22-49-24.toml --model benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_salicylic_19-Nov-2025_22-49-24/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_toluene_19-Nov-2025_22-49-24/md_toluene_19-Nov-2025_22-49-24.toml --model benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_toluene_19-Nov-2025_22-49-24/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```

```bash
uv run equivariance_defect --config benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_uracil_19-Nov-2025_22-49-24/md_uracil_19-Nov-2025_22-49-24.toml --model benchmark_runs/md17_uniform_paper_atom_non_equivariant_19-Nov-2025_22-49-24/md_uracil_19-Nov-2025_22-49-24/run_1/best_val_model.pth --num_rotations 20 --rotation_seed 42
```


</summmary>
</details>

## Pre-trained Models

You can download pretrained models here:

- [Anonymized for review]()

## TG80 Dataset

- [Anonymized for review]()

## Results

Our model achieves the following performance:

### Single-task Trajectory Prediction on [MD17](https://www.sgdml.org/)
![MD17_ST_Results](Z_paper_content/readme_content/md17_results.png)

### Multitask Trajectory Position Prediction on [TG80]()
![MD17_ST_Results](Z_paper_content/readme_content/tg80_results.png)


## Further notes
The notation in the paper generally corresponds to our comments, with the following caveats:
* Timesteps - P -> T

## Contributing
Both ATOM and TG80 are under the MIT licence.

