import json
from pathlib import Path

import pytest

from analysis_code import cli


class DummyArgs:
    """Simple stand‑in for argparse.Namespace for CLI helpers."""

    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_tables_mse_writes_expected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_tables('mse') should call build_tables_from_dirs and write into PAPER_ROOT/tables."""

    # Redirect PAPER_ROOT to a temporary test directory
    monkeypatch.setattr(cli, "PAPER_ROOT", tmp_path / "paper_root")

    # Create fake run directories
    run1: Path = tmp_path / "runs" / "egno"
    run2: Path = tmp_path / "runs" / "atom"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)

    called: dict[str, object] = {}

    def fake_build_tables_from_dirs(dirs: list[Path], bold_best: bool) -> dict[str, str]:
        called["dirs"] = dirs
        called["bold_best"] = bold_best
        return {"uniform": "MSE_UNIFORM_LATEX"}

    def fake_collect_results(dirs: list[Path]) -> list[object]:
        return ["dummy"]

    def fake_infer_token(results: list[object]) -> str:
        assert results == ["dummy"]
        return "md17"

    monkeypatch.setattr(
        "Z_paper_content.create_mse_tables.build_tables_from_dirs",
        fake_build_tables_from_dirs,
    )
    monkeypatch.setattr(
        "Z_paper_content.create_mse_tables._collect_results_from_dirs",
        fake_collect_results,
    )
    monkeypatch.setattr(
        "Z_paper_content.create_mse_tables._infer_dataset_token_from_results",
        fake_infer_token,
    )

    args = DummyArgs(command="tables", subcommand="mse", dirs=[str(run1), str(run2)], no_bold=False)
    ret: int = cli.cmd_tables(args)  # type: ignore[arg-type]
    assert ret == 0

    # Verify underlying helper was called with resolved paths
    assert called["bold_best"] is True
    passed_dirs: list[Path] = called["dirs"]  # type: ignore[assignment]
    assert run1.resolve() in passed_dirs and run2.resolve() in passed_dirs

    tables_dir: Path = tmp_path / "paper_root" / "tables"
    expected_file: Path = tables_dir / "md17_uniform_tables.tex"
    alias_file: Path = tables_dir / "md17_tables.tex"
    assert expected_file.read_text(encoding="utf-8") == "MSE_UNIFORM_LATEX"
    assert alias_file.read_text(encoding="utf-8") == "MSE_UNIFORM_LATEX"


def test_tables_runtime_writes_runtime_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_tables('runtime') should call build_runtime_table and write runtime_*.tex."""

    monkeypatch.setattr(cli, "PAPER_ROOT", tmp_path / "paper_root")

    egno_dir: Path = tmp_path / "egno"
    atom_dir: Path = tmp_path / "atom"
    egno_dir.mkdir()
    atom_dir.mkdir()

    called: dict[str, object] = {}

    def fake_build_runtime_table(egno_dir: Path, atom_dir: Path, dataset: str, f_peak_tflops: float) -> str:
        called["egno_dir"] = egno_dir
        called["atom_dir"] = atom_dir
        called["dataset"] = dataset
        called["f_peak"] = f_peak_tflops
        return "RUNTIME_LATEX"

    monkeypatch.setattr(
        "Z_paper_content.create_runtime_tables.build_runtime_table",
        fake_build_runtime_table,
    )

    args = DummyArgs(
        command="tables",
        subcommand="runtime",
        egno_dir=str(egno_dir),
        atom_dir=str(atom_dir),
        dataset="md17",
        f_peak=15.0,
    )
    ret: int = cli.cmd_tables(args)  # type: ignore[arg-type]
    assert ret == 0

    assert called["dataset"] == "md17"
    assert called["f_peak"] == 15.0
    assert called["egno_dir"] == egno_dir.resolve()
    assert called["atom_dir"] == atom_dir.resolve()

    runtime_file: Path = tmp_path / "paper_root" / "tables" / "runtime_md17.tex"
    assert runtime_file.read_text(encoding="utf-8") == "RUNTIME_LATEX"


def test_tables_all_delegates_to_run_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_tables('all') should delegate to run_tables.run_tables_variadic with correct args."""

    called: dict[str, object] = {}

    def fake_run_tables_variadic(dirs: list[str], bold_best: bool) -> int:
        called["dirs"] = dirs
        called["bold_best"] = bold_best
        return 0

    monkeypatch.setattr(
        "Z_paper_content.run_tables.run_tables_variadic",
        fake_run_tables_variadic,
    )

    args = DummyArgs(
        command="tables",
        subcommand="all",
        dirs=["/path/to/exp1", "/path/to/exp2"],
        no_bold=True,
    )
    ret: int = cli.cmd_tables(args)  # type: ignore[arg-type]
    assert ret == 0
    assert called["dirs"] == ["/path/to/exp1", "/path/to/exp2"]
    assert called["bold_best"] is False  # no_bold=True => bold_best=False


def test_figures_ablations_uses_default_output_under_paper_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_figures('ablations') should default output into PAPER_ROOT/ablations when no --output is given."""

    monkeypatch.setattr(cli, "PAPER_ROOT", tmp_path / "paper_root")

    ablation_dir: Path = tmp_path / "abl_runs"
    ablation_dir.mkdir()

    def fake_set_style(*_: object, **__: object) -> None:
        return None

    captured: dict[str, object] = {}

    def fake_plot_ablations(ablation_dir: Path, save_path: Path, error_bar_type: object, add_text: bool) -> None:
        captured["ablation_dir"] = ablation_dir
        captured["save_path"] = save_path
        captured["error_bar_type"] = error_bar_type
        captured["add_text"] = add_text

    monkeypatch.setattr("Z_paper_content.figures.set_matplotlib_style", fake_set_style)
    monkeypatch.setattr("Z_paper_content.ablations.plot_ablations", fake_plot_ablations)

    args = DummyArgs(
        command="figures",
        subcommand="ablations",
        ablation_dir=str(ablation_dir),
        output=None,
        error_bar_type="percentile",
        add_text=True,
    )
    ret: int = cli.cmd_figures(args)  # type: ignore[arg-type]
    assert ret == 0

    assert captured["ablation_dir"] == ablation_dir.resolve()
    save_path: Path = captured["save_path"]  # type: ignore[assignment]
    assert save_path == tmp_path / "paper_root" / "ablations" / "ablation_MD17_ST.pdf"


def test_figures_invariance_p_delegates_to_run_p_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_figures('invariance-p') should parse p and call run_p_invariance."""

    from types import SimpleNamespace

    called: dict[str, object] = {}

    def fake_parse_numeric_list(arg: str | list[int] | None) -> list[int]:
        assert arg == "[4,8]"
        return [4, 8]

    def fake_run_p_invariance(
        p_values: list[int],
        config_paths: list[str] | str,
        model_paths: list[str] | str,
        save_dir: str | None = None,
    ) -> dict[str, list[tuple[int, float, float]]]:
        called["p_values"] = p_values
        called["config_paths"] = config_paths
        called["model_paths"] = model_paths
        called["save_dir"] = save_dir
        return {}

    monkeypatch.setattr("Z_paper_content.invariances._parse_numeric_list", fake_parse_numeric_list)
    monkeypatch.setattr("Z_paper_content.invariances.run_p_invariance", fake_run_p_invariance)

    args = DummyArgs(
        command="figures",
        subcommand="invariance-p",
        p="[4,8]",
        config=["cfg1.toml", "cfg2.toml"],
        model=["runs1", "runs2"],
        save_dir="out_dir",
    )
    ret: int = cli.cmd_figures(args)  # type: ignore[arg-type]
    assert ret == 0
    assert called["p_values"] == [4, 8]
    assert called["config_paths"] == ["cfg1.toml", "cfg2.toml"]
    assert called["model_paths"] == ["runs1", "runs2"]
    assert called["save_dir"] == "out_dir"


def test_figures_invariance_t_delegates_to_run_t_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_figures('invariance-t') should parse t and call run_t_invariance."""

    called: dict[str, object] = {}

    def fake_parse_numeric_list(arg: str | list[int] | None) -> list[int]:
        assert arg == "[1,2]"
        return [1, 2]

    def fake_run_t_invariance(
        t_values: list[int],
        config_paths: list[str] | str,
        model_paths: list[str] | str,
        save_dir: str | None = None,
    ) -> dict[str, list[tuple[int, float, float]]]:
        called["t_values"] = t_values
        called["config_paths"] = config_paths
        called["model_paths"] = model_paths
        called["save_dir"] = save_dir
        return {}

    monkeypatch.setattr("Z_paper_content.invariances._parse_numeric_list", fake_parse_numeric_list)
    monkeypatch.setattr("Z_paper_content.invariances.run_t_invariance", fake_run_t_invariance)

    args = DummyArgs(
        command="figures",
        subcommand="invariance-t",
        t="[1,2]",
        config=["cfg.toml"],
        model=["runs"],
        save_dir=None,
    )
    ret: int = cli.cmd_figures(args)  # type: ignore[arg-type]
    assert ret == 0
    assert called["t_values"] == [1, 2]
    assert called["config_paths"] == ["cfg.toml"]
    assert called["model_paths"] == ["runs"]
    assert called["save_dir"] is None


def test_analyze_dataset_uses_global_max_and_calls_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_analyze('dataset') should compute a max over .npz files and call the helper once."""

    from Z_paper_content import dataset_info
    import numpy as np

    data_dir: Path = tmp_path / "md17_npz"
    data_dir.mkdir()

    # Create a simple R array with shape (T, N, 3)
    arr = np.zeros((2, 1, 3), dtype=float)
    arr[1, 0, 0] = 2.0  # variance > 0
    np.savez(data_dir / "md17_dummy.npz", R=arr)

    called: dict[str, object] = {}

    def fake_create_vis(data_dir_arg: Path, dataset_name: str, global_max_x: float) -> None:
        called["data_dir"] = data_dir_arg
        called["dataset_name"] = dataset_name
        called["global_max_x"] = global_max_x

    monkeypatch.setattr(dataset_info, "create_corrected_volatility_visualization", fake_create_vis)

    args = DummyArgs(
        command="analyze",
        subcommand="dataset",
        data_dir=str(data_dir),
        dataset="md17",
    )
    ret: int = cli.cmd_analyze(args)  # type: ignore[arg-type]
    assert ret == 0
    assert called["data_dir"] == data_dir.resolve()
    assert called["dataset_name"] == "md17"
    assert called["global_max_x"] > 0.0


def test_analyze_timing_writes_default_json_when_output_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_analyze('timing') should call run_experiment and write JSON under PAPER_ROOT by default."""

    from Z_paper_content import md17_md_vs_atom_timing as timing_mod

    monkeypatch.setattr(cli, "PAPER_ROOT", tmp_path / "paper_root")

    atom_run_dir: Path = tmp_path / "atom_runs"
    atom_run_dir.mkdir()

    def fake_run_experiment(atom_run_dir: Path, num_repeats: int, limit_molecules: list[str] | None) -> dict[str, object]:
        assert atom_run_dir == atom_run_dir.resolve()
        assert num_repeats == 2
        assert limit_molecules == ["aspirin"]
        return {"ok": True}

    monkeypatch.setattr(timing_mod, "run_experiment", fake_run_experiment)

    args = DummyArgs(
        command="analyze",
        subcommand="timing",
        atom_run_dir=str(atom_run_dir),
        num_repeats=2,
        molecules=["aspirin"],
        output_json=None,
    )
    ret: int = cli.cmd_analyze(args)  # type: ignore[arg-type]
    assert ret == 0

    expected_json: Path = tmp_path / "paper_root" / "md17_md_vs_atom_timing.json"
    data = json.loads(expected_json.read_text(encoding="utf-8"))
    assert data == {"ok": True}


