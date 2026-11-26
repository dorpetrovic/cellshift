# CellShift

**Do gene-knockdown predictors know when they are looking at a cell type they have never seen?**

CellShift evaluates whether CRISPRi knockdown-effect predictors (ridge, kNN, MLP ensembles on
frozen ESM-2 protein embeddings) can be trusted when moved to a new cell line. Four lines
(K562, RPE1, HepG2, Jurkat), leave-one-cell-line-out, four questions:

1. **Accuracy drop:** how much worse is each model on a line it never saw?
2. **Calibration:** do 90% conformal and ensemble intervals still cover 90% there?
3. **Warning score:** can a label-free signal (embedding distance, ensemble disagreement) flag the big errors?
4. **Cheap fixes:** how much do few-shot recalibration (20/50/100 knockdowns), weighted conformal or cell-context features recover?

The shift tools live in `cellshift.shift_eval` and are domain-agnostic, so they can be reused on other benchmarks.

Every number is reported with a 95% bootstrap interval over knockdowns.

> Status: v0.2 in progress, module 1 of 4.

## Splits

| Split | Train | Test |
| --- | --- | --- |
| Unseen knockdown | one line, most response groups | same line, held-out groups |
| Unseen line | three lines | fourth line, knockdowns seen in training |
| Unseen both | three lines, most knockdowns | fourth line, held-out knockdowns |

Each split is checked by `cellshift.splits.assert_no_leakage`, and CI fails on any overlap.

## Quick start

```bash
pip install -e ".[dev]" && pytest -q      # core library + tests, no data needed
make smoke                                # K562 end to end (downloads data)
make install && make reproduce            # all four modules
```

## Related work

- Yanovsky and Chernomoretz, *When perturbation queries fail: static gene and protein embeddings outperform simulated scFM perturbation responses*, NeurIPS 2026 ICBINB workshop. Within-line results; CellShift extends to cross-line splits and calibrated uncertainty.
- Arc Institute, Virtual Cell Challenge 2026: zero-shot CRISPRi response prediction in unseen cell lines.

## Data

Replogle et al. 2022 (K562 essential, RPE1) via pertpy; HepG2 and Jurkat from GEO GSE264667.

## License

MIT
