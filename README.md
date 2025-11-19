# ATOM: A Pretrained Neural Operator for Multitask Dynamics Learning

This repository is the official implementation of [ATOM: A Pretrained Neural Operator for Multitask Dynamics Learning](PLACEHOLDER). ATOM is a graph transformer neural operator for the parallel decoding of molecular dynamics trajectory. We show state-of-the-art performance on the existing MD17 dataset, and for the first time, demonstrate zero-shot generalization to unseen chemical compounds.

![ATOM Diagram](Z_paper_content/readme_content/arch.png)

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
<summary>Evaluating Equivariance Error on ATOM</summary>

### Evaluating baseline, equivariant lifting, fully equivariant
```bash
uv run python /home/luke/gtno_py/tests/test_equivariance.py /home/luke/gtno_py/benchmark_runs/equivariance_error
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

### [Single-task Trajectory Prediction on MD17](https://www.sgdml.org/)
![MD17_ST_Results](Z_paper_content/readme_content/md17_results.png)

### [Multitask Trajectory Position Prediction on TG80]()
![MD17_ST_Results](Z_paper_content/readme_content/tg80_results.png)


## Further notes
The notation in the paper generally corresponds to our comments, with the following caveats:
* Timesteps - P -> T

## Contributing
Both ATOM and TG80 are under the MIT licence.

