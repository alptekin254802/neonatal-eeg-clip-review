# Selecting neonatal electroencephalogram clips for review beyond the first reader’s seizure marks

Alptekin Cinbat, Ümit Şentürk and İbrahim Yücedağ

A pretrained convolutional neural network ranked 30-second clips outside an expert’s seizure marks in 79 Helsinki neonatal EEG recordings. In the held-out group, mean capture across the 13 infants with targets was 41.1%, compared with 11.1% expected under uniform random selection at the same clip budget. These outcomes measure the EEG displayed for reassessment.

## Contents

| Directory | Contents |
| --- | --- |
| `src/` | Data acquisition, EEG preprocessing, pretrained inference, clip selection and statistical analysis |
| `reported/` | Aggregate results from Tables 1 and 2, at manuscript precision |

## Installation

Use 64-bit Python 3.11 and MATLAB R2025b with Signal Processing Toolbox. The `matlab` executable must be on PATH. From the repository root in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

The Helsinki neonatal EEG dataset, version 4, contains 79 EDF recordings and three sets of second-by-second expert annotations. Download the data, published model and channel annotations before running the analysis. The EDF downloads total approximately 4.3 GB. [Helsinki data](https://doi.org/10.5281/zenodo.4940267) · [Isaev model and weights](https://github.com/dyisaev/seizure-detection-neonates) · [Channel annotations and signal preprocessing](https://github.com/ktapani/Neonatal_Seizure_Detection)

```powershell
python src/acquire.py --data data --reference-example
```

The download checks the EDF and annotation files against the checksums in the version 4 Zenodo record. It also retrieves the original model source and weights, three MATLAB preprocessing files, the channel annotation workbook and the authors’ EEG1 example. Allow roughly 10 GB for data and intermediate files, plus the Python environment. Download time depends on bandwidth. Repeating the command reuses verified downloads.

## Reproduction

```powershell
python src/score.py --data data --output scores --check-preprocessing
python src/score.py --data data --output scores
python src/evaluate.py --data data --scores scores --output results
python src/compare_tables.py --results results
```

| Step | Output | Expected result and approximate time |
| --- | --- | --- |
| Check the preprocessing example | `scores/preprocessing_check.json` | 1,747 epochs, 1,522 valid, zero flag mismatches. Allow a minute for MATLAB startup and comparison. |
| Score the recordings | `scores/1.npz` through `scores/79.npz` | One score per second from overlapping model windows. Allow several minutes on a desktop CPU, including MATLAB preprocessing. |
| Select clips and estimate capture | `results/table_1.csv`, `table_2.csv`, their Markdown copies, and `results/summary.json` | About five seconds after scoring. Table 1 gives capture by method. Table 2 gives the amount of EEG selected and the fraction of clips overlapping a target. |
| Compare both tables | Console | Checks row labels and all 40 cohort-specific entries against the reported tables at manuscript precision. Under one second. |

The summary also gives target duration, the effect of omitting each infant on the range of the estimate, the combined estimate across the two groups, and overlap with the published channel annotations. The primary comparison gives equal weight to the 13 held-out infants with targets. The count of 29 captured instances out of 90 pools events across these infants. The 38 target-overlapping clips out of 767 selected use all 117 simulated readings.

## Citation

Cinbat A, Şentürk Ü, Yücedağ İ. Selecting neonatal electroencephalogram clips for review beyond the first reader’s seizure marks. Manuscript.

Please also cite [Stevenson et al.](https://doi.org/10.1038/sdata.2019.39) for the data and [Isaev et al.](https://proceedings.mlr.press/v126/isaev20a.html) for the pretrained model.

## License

The authors’ analysis software is available under the [MIT License](LICENSE). The documentation and aggregate tables in `reported/` are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([license text](LICENSE-CC-BY-4.0.txt)). Credit Alptekin Cinbat, Ümit Şentürk and İbrahim Yücedağ and cite the manuscript when reusing those materials. The recordings, expert annotations, third-party source code, pretrained weights and reference example are supplied by their respective original providers under their own terms.
