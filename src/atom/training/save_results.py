from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, computed_field
import torch

from atom.training.create_config import Config


class SingleRunResults(BaseModel):
    s2t_test_loss: float
    s2s_test_loss: float
    best_val_loss_epoch: int
    start_time: datetime
    end_time: datetime
    model_path: Path

    @computed_field
    @property
    def run_time(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @computed_field
    @property
    def seconds_per_epoch(self) -> float | None:
        if self.run_time and self.best_val_loss_epoch > 0:
            return self.run_time / self.best_val_loss_epoch
        return None


class MultiRunResults(BaseModel):
    single_run_results: list[SingleRunResults]
    config: Config

    @computed_field
    @property
    def s2s_test_loss_mean(self) -> float:
        return sum(result.s2s_test_loss for result in self.single_run_results) / len(self.single_run_results)

    @computed_field
    @property
    def s2s_test_loss_std(self) -> float:
        return torch.std(torch.tensor([result.s2s_test_loss for result in self.single_run_results])).item()

    @computed_field
    @property
    def s2s_test_loss_max(self) -> float:
        return max(result.s2s_test_loss for result in self.single_run_results)

    @computed_field
    @property
    def s2s_test_loss_min(self) -> float:
        return min(result.s2s_test_loss for result in self.single_run_results)

    @computed_field
    @property
    def latex_s2s(self) -> str:
        return f"\\({self.s2s_test_loss_mean * 100:.2f}{{\\scriptstyle \\pm{self.s2s_test_loss_std * 100:.2f}}}\\)"

    @computed_field
    @property
    def s2t_test_loss_mean(self) -> float:
        return sum(result.s2t_test_loss for result in self.single_run_results) / len(self.single_run_results)

    @computed_field
    @property
    def s2t_test_loss_std(self) -> float:
        return torch.std(torch.tensor([result.s2t_test_loss for result in self.single_run_results])).item()

    @computed_field
    @property
    def s2t_test_loss_max(self) -> float:
        return max(result.s2t_test_loss for result in self.single_run_results)

    @computed_field
    @property
    def s2t_test_loss_min(self) -> float:
        return min(result.s2t_test_loss for result in self.single_run_results)

    @computed_field
    @property
    def latex_s2t(self) -> str:
        return f"\\({self.s2t_test_loss_mean * 100:.2f}{{\\scriptstyle \\pm{self.s2t_test_loss_std * 100:.2f}}}\\)"

    @computed_field
    @property
    def mean_secs_per_run(self) -> float:
        return sum(result.run_time for result in self.single_run_results) / len(self.single_run_results)

    @computed_field
    @property
    def mean_secs_per_epoch(self) -> float:
        return sum(result.seconds_per_epoch for result in self.single_run_results if result.seconds_per_epoch is not None) / len(
            self.single_run_results
        )

    @computed_field
    @property
    def mean_best_val_loss_epoch(self) -> float:
        return sum(result.best_val_loss_epoch for result in self.single_run_results) / len(self.single_run_results)
