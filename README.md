# Subjective label properties across rater groups

Analysis code for a case study comparing the reliability and structure of subjective labels for identical images across rater groups, using outfit quirkiness ratings by Japanese and Vietnamese raters.

The paper is in preparation. A citation will be added when it is available.

## What the code computes

Each score is decomposed within a set and a rater group as

```
y_ri = m + c_i + a_r + e_ri
```

where `m` is the set mean, `c_i` the image consensus, `a_r` the rater level, and `e_ri` the residual.

| Quantity | Meaning |
|---|---|
| `rho` | ICC(C,1): agreement after removing rater levels |
| Required raters | Raters needed for a target reliability, by the Spearman–Brown formula |
| `H`, `H_res` | Mean correlation over the 15 pairs of six elements, without and with the image consensus removed |
| `rho*` | Rank correlation of the two groups' consensuses, corrected for reliability |
| Model fit | Corrected rank correlation between a model trained on separate Japanese ratings and each group's consensus |

The permutation test exchanges group labels between raters with the images and the trained model held fixed, and is Holm-adjusted over nine comparisons. The intervals also resample images, and for the model, the training data, so the two can lead to different conclusions.

## Usage

The anonymised ratings and model predictions are in `data/`; see [data/README.md](data/README.md).

Image features are not released, so the code uses the released per-image model predictions. Three values cannot be recomputed from the released data and are read from a fixed file: the own-group model comparison, the transfer loss, and the model gap interval. Model training itself cannot be rerun.

```
pip install -r requirements.txt
python analysis/reproduce.py          # full run, several minutes or more
python analysis/reproduce.py --quick  # short check of the pipeline
```

Results are written to `results/summary.md`.
