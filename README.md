# ATOM: A Pretrained Neural Operator for Multitask Dynamics Learning

This repository is the official implementation of [ATOM: A Pretrained Neural Operator for Multitask Dynamics Learning](https://arxiv.org/abs/2030.12345). ATOM is a graph transformer neural operator for the parallel decoding of molecular dynamics trajectory. We show state-of-the-art performance on existing datasets, and for the first time, demonstrate zero-shot generalisation to unseen chemical compounds.

![ATOM Diagram](Z_paper_content/ATOM%20Architecture.png)

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

For example, to evaluate the performance when Δt = 3000, run:

```bash
poetry run inference --model benchmark_runs/t_invariance/delta_t_3000_aspirin_13-Apr-2025_01-46-23/run_3/best_val_model.pth --config configs/t_invariance/3000.toml
```

To test equivariance run:
```bash
python tests/test_equivariance.py --config configs/md17_paper/md_aspirin.toml --model benchmark_runs/paper_md17_singletask_12-May-2025_23-33-44/run_1/best_val_model.pth
```

To test the zero-shot generalization ability for each of the five molecules run the following commands:
Uracil
```bash
poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold1_multitask_15-May-2025_09-36-35/run_1/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold1.toml

poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold1_multitask_15-May-2025_12-39-07/run_1/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold1.toml

poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold1_multitask_15-May-2025_12-39-07/run_2/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold1.toml
```

Nitrobenzene
```bash
poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold2_multitask_15-May-2025_10-31-21/run_1/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold2.toml

poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold2_multitask_15-May-2025_13-18-52/run_1/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold2.toml

poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold2_multitask_15-May-2025_13-18-52/run_2/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold2.toml
```

Indole
```bash
poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold3_multitask_15-May-2025_11-20-46/run_1/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold3.toml

poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold3_multitask_15-May-2025_14-10-00/run_1/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold3.toml

poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold3_multitask_15-May-2025_14-10-00/run_2/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold3.toml
```

Napthalene
```bash
poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold4_multitask_15-May-2025_12-08-25/run_1/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold4.toml

poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold4_multitask_15-May-2025_12-39-39/run_1/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold4.toml

poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold4_multitask_15-May-2025_12-39-39/run_2/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold4.toml
```

Butanol
```bash
poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold5_multitask_15-May-2025_12-40-10/run_1/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold5.toml

poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold5_multitask_15-May-2025_12-40-10/run_2/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold5.toml

# poetry run inference --model benchmark_runs/tg80_atom_mt/atom_tg80_multitask_muon_fold5_multitask_15-May-2025_12-40-10/run_2/best_val_model.pth --config configs/tg80_multitask/atom_multitask_muon_fold5.toml
```
You may wish to confirm that none of these pretrained multitask models were trained on the molecules for which we inference them by inspecting their config files.

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Further notes
The notation in the paper generally corresponds to our comments, with the following caveats:
* Timesteps - P -> T
* GTNO (an earlier name) -> ATOM

## Contributing

MIT License

