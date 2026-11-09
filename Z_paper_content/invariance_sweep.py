"""Evaluate trained checkpoints across P or Delta T and store the S2T/S2S MSE per seed.

Usage:
  python -m Z_paper_content.invariance_sweep --kind p --values 4 6 8 12 16 24 \
      --series ATOM=<benchmark_dir> EGNO=<benchmark_dir> --out results.json

Each <benchmark_dir> holds the run's .toml and run_*/best_val_model.pth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Z_paper_content.invariances import _run_single_eval
from atom.training import Config


def _find(bench_dir: Path) -> tuple[Path, list[Path]]:
    tomls = sorted(p for p in bench_dir.glob("*.toml"))
    if len(tomls) != 1:
        raise FileNotFoundError(f"expected one toml in {bench_dir}, found {tomls}")
    ckpts = sorted(bench_dir.glob("run_*/best_val_model.pth"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {bench_dir}")
    return tomls[0], ckpts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["p", "t"], required=True)
    ap.add_argument("--values", type=int, nargs="+", required=True)
    ap.add_argument("--series", nargs="+", required=True, help="LABEL=benchmark_dir")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out: dict[str, dict] = {"kind": args.kind, "values": args.values, "series": {}}
    for spec in args.series:
        label, d = spec.split("=", 1)
        cfg_path, ckpts = _find(Path(d))
        rows = []
        for v in args.values:
            s2t, s2s = [], []
            for ck in ckpts:
                cfg = Config.from_toml(cfg_path)
                if args.kind == "p":
                    cfg.dataloader.num_timesteps = int(v)
                else:
                    cfg.dataloader.delta_T = int(v)
                a, b = _run_single_eval(cfg, ck)
                s2t.append(a)
                s2s.append(b)
            rows.append({"value": v, "s2t": s2t, "s2s": s2s})
            print(label, v, [round(x * 100, 3) for x in s2t], flush=True)
        out["series"][label] = {"source": str(d), "rows": rows}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
