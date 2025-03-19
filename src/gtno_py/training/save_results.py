from pydantic import BaseModel, model_validator
from datetime import datetime
from pathlib import Path
from gtno_py.training import Config
import torch


class SingleRunResults(BaseModel):
    s2t_test_loss: float
    s2s_test_loss: float
    best_val_loss_epoch: int
    run_time: float
    seconds_per_epoch: float
    start_time: datetime
    end_time: datetime
    model_path: Path


class MultiRunResults(BaseModel):
    single_run_results: list[SingleRunResults]

    s2s_test_loss_mean: float | None = None
    s2s_test_loss_std: float | None = None
    s2s_test_loss_max: float | None = None
    s2s_test_loss_min: float | None = None
    latex_s2s: str | None = None

    s2t_test_loss_mean: float | None = None
    s2t_test_loss_std: float | None = None
    s2t_test_loss_max: float | None = None
    s2t_test_loss_min: float | None = None
    latex_s2t: str | None = None

    mean_secs_per_run: float | None = None
    mean_secs_per_epoch: float | None = None

    mean_best_val_loss_epoch: float | None = None

    config: Config

    @model_validator(mode="after")
    def construct_multi_run_results(self) -> "MultiRunResults":
        self.s2s_test_loss_mean = sum(result.s2s_test_loss for result in self.single_run_results) / len(self.single_run_results)
        self.s2s_test_loss_std = torch.std(torch.tensor([result.s2s_test_loss for result in self.single_run_results])).item()
        self.s2s_test_loss_max = max(result.s2s_test_loss for result in self.single_run_results)
        self.s2s_test_loss_min = min(result.s2s_test_loss for result in self.single_run_results)

        self.s2t_test_loss_mean = sum(result.s2t_test_loss for result in self.single_run_results) / len(self.single_run_results)
        self.s2t_test_loss_std = torch.std(torch.tensor([result.s2t_test_loss for result in self.single_run_results])).item()
        self.s2t_test_loss_max = max(result.s2t_test_loss for result in self.single_run_results)
        self.s2t_test_loss_min = min(result.s2t_test_loss for result in self.single_run_results)

        self.mean_secs_per_run = sum(result.run_time for result in self.single_run_results) / len(self.single_run_results)
        self.mean_secs_per_epoch = sum(result.seconds_per_epoch for result in self.single_run_results) / len(self.single_run_results)

        self.mean_best_val_loss_epoch = sum(result.best_val_loss_epoch for result in self.single_run_results) / len(self.single_run_results)

        return self

    @model_validator(mode="after")
    def compute_latex(self) -> "MultiRunResults":
        if self.s2s_test_loss_mean is not None:
            self.latex_s2s = f"\\({self.s2s_test_loss_mean*100:.2f}{{\\scriptstyle \\pm{self.s2s_test_loss_std*100:.2f}}}\\)"
        if self.s2t_test_loss_mean is not None:
            self.latex_s2t = f"\\({self.s2t_test_loss_mean*100:.2f}{{\\scriptstyle \\pm{self.s2t_test_loss_std*100:.2f}}}\\)"
        return self
