# ATOM repository guide for agents

ATOM is a graph transformer neural operator for molecular dynamics trajectory
prediction (MD17, rMD17, TG80). Code is in `src/`, configs in `configs/`,
analysis and figure scripts in `analysis_code/` and `Z_paper_content/`, and
results in `results/` and `benchmark_runs/`.

## The paper (`paper/`)

`paper/` holds the LaTeX source of the ATOM paper. It is a **git subtree** of
the Overleaf project `67582be27ea98088ba09e7c6`, so it is ordinary files in this
repo and you can read, grep and edit it like any other directory.

### Layout

| File | Contents |
|---|---|
| `paper/content.tex` | Main body: Introduction, Related work, ATOM method (problem formulation, model and training design, multitask pretraining / TG80), Experiments (single-task, multitask on TG80, temporal-gap / timestep invariance, ablations), Conclusions |
| `paper/appendices_temp.tex` | Appendices: group background, datasets (licences, dataloader, numerical stability, TG80 generation, MD simulations), architectural details (equivariant ATOM, RWPE, value-residual, delta-prediction), further experiments, experimental details (hardware, training/inference times, ATOM and EGNO hyperparameters), proofs, trajectory samples |
| `paper/acronyms.tex` | `\gls` acronym definitions |
| `paper/library.bib` | Bibliography |
| `paper/Figures/` | Figures included by the `.tex` files |
| `paper/ICLR2026/iclr2026_conference.tex` | ICLR 2026 main document (`\input`s acronyms, content, appendices_temp) |
| `paper/NeurIPS 2025/arxiv_submission.tex` | arXiv/NeurIPS main document (same inputs) |
| `paper/DO NOT USE PrimeArxiv Template/` | Old template and proposal: ignore |

Paths inside the `.tex` files are relative to `paper/` (Overleaf's project
root), e.g. `Figures/...` and `ICLR2026/iclr2026_conference`.

### Cross-referencing code and paper

- Find a section: `grep -n '\\section\|\\subsection' paper/content.tex paper/appendices_temp.tex`
- Find a label, figure or table: `grep -rn 'label{tab:' paper/*.tex`
- Hyperparameter tables in the appendix ("ATOM Hyperparameters") should match
  `configs/`; figures in `paper/Figures/` are produced by
  `Z_paper_content/figures.py`, `analysis_code/` and the table scripts in
  `Z_paper_content/`. When code changes alter reported numbers, say which
  table or figure in the paper is now stale.

### Keeping it in sync with Overleaf

```bash
scripts/paper_sync.sh pull   # Overleaf -> paper/
scripts/paper_sync.sh push   # paper/ commits -> Overleaf
```

- Pull before editing `paper/`, and before relying on its contents for
  anything time-sensitive; Overleaf is the source of truth.
- Keep commits that touch `paper/` separate from code commits, so
  `git subtree push` sends clean history to Overleaf.
- Do not push to Overleaf unless the user asks: it changes the shared paper
  immediately.
- Authentication: user `git`, password an Overleaf token. Never write the
  token into the repo, a remote URL or a commit. In cloud sessions use the
  "Overleaf paper" environment, whose proxy injects it; locally,
  `export OVERLEAF_TOKEN=...` or use a git credential helper.
- Do not edit `paper/README.md` for repo docs: it is part of the Overleaf
  project. Repo-level docs go here.
