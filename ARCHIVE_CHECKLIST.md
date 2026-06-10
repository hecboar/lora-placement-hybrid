# Archive / Submission Checklist

Pre-submission and Zenodo-archival checklist for *"Where Should LoRA Go?"*

- [x] README updated to v4 framing
- [x] REPRODUCIBILITY.md added
- [x] DATA_AVAILABILITY.md added
- [x] CITATION.cff added
- [x] requirements/environment file added
- [x] local paths removed or documented
- [x] HumanEval floor effect documented
- [x] bootstrap settings documented
- [ ] exact model revisions added (`Qwen/Qwen3.5-0.8B-Base`, `tiiuae/Falcon-H1-0.5B-Base` HF commit hashes)
- [x] core package versions pinned from `environment_snapshot.json` (torch/transformers/peft/trl/datasets; Python 3.11.10) — secondary deps (accelerate, bitsandbytes, numpy, pandas, scipy, scikit-learn, matplotlib) still to pin
- [ ] GitHub release created
- [ ] Zenodo DOI created
- [ ] DOI and commit hash inserted into manuscript
- [ ] DOI and commit hash inserted into README
- [ ] DOI inserted into CITATION.cff
- [ ] repository archived

## Remaining placeholders to fill before submission

| Location | Placeholder |
|----------|-------------|
| `README.md` (Reproducibility, Citation) | Zenodo DOI, Git commit |
| `REPRODUCIBILITY.md` (Base models) | exact HF revisions; exact package versions |
| `DATA_AVAILABILITY.md` (Archival identifiers) | Zenodo DOI, Git commit |
| `CITATION.cff` (`doi:`) | Zenodo DOI |
| `requirements.txt` / `environment.yml` | exact pinned versions |
