# Cross-Room Few-Shot WiFi Activity Recognition (CS60055 Challenge 2)

A Python pipeline that goes from the raw challenge package (Room A Nexmon `.pcap`, Room B amplitude `.csv`) to 150 × 256 CSI amplitude windows. It then runs three tasks on those windows:

1. **Task 1:** a room fingerprint/router (Room A vs Room B).
2. **Task 2:** a self-supervised CSI encoder, evaluated with linear probes.
3. **Task 3:** cross-room few-label activity adaptation for K ∈ {1, 5, 10, 20} in both directions.

All numbers reported here and in [report/report.md](report/report.md) are **validation** results (not unseen-test results).

## Setup

```bash
# Python 3.11 (CPU is enough; the whole pipeline was developed on a laptop CPU)
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
# (plain pip works too: python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu)
```

## Data

The dataset is **not** part of the repository. Download the participant package from the challenge Google Drive into `data/raw/`:

```bash
python scripts/download_data.py --out data/raw
```

The script mirrors the unpacked `Experiment-2/` and `Experiment-3/` folders. It deliberately skips the `Experiment-*.zip` archives in the same Drive folder, because those archives bundle the source authors' MATLAB code, results, and LSTM notebook, none of which may be used. If you place the data elsewhere, edit `paths:` in [configs/experiment.yaml](configs/experiment.yaml).

Expected counts (checked by `scripts/inspect_data.py`):

- **Room A:** 1,084 pcaps (934 development + 150 validation, 25 per class).
- **Room B:** 1,201 CSVs (including the four non-`_Amp_` files `Empty_11`, `Lying_0`, `Sitting_0`, `Standing_0`; `model_plot_1.png` is ignored).

## Reproduce everything

```bash
python run_all.py            # inspect -> dataset -> ssl -> task1 -> task2 -> task3 -> figures
python run_all.py --from task1   # resume from a stage
```

| Stage | Script | Output |
|---|---|---|
| inspect | `scripts/inspect_data.py` | `results/data_inspection.json` (file types, packet/payload structure, header fields, sequence/timestamp behaviour, CSV schema) |
| dataset | `scripts/build_dataset.py` | `workspace/intermediate_csv/room_{A,B}/<capture>.csv` (256 amplitude columns `subcarrier_001..256`), `workspace/manifest.csv` (capture provenance, QC counts), `workspace/windows/{development,local_validation}/window_XXXXXX.npy` (150 × 256 float32), `workspace/windows/index.csv` (window → capture, rows used, method, config hash) |
| ssl | `scripts/train_ssl.py` | `workspace/artifacts/{normalizer.json, encoder_ssl.pt, repr_*.npz}` |
| task1 | `scripts/task1_room.py` | `results/task1_*.csv`, `workspace/artifacts/router.pkl` |
| task2 | `scripts/task2_ssl.py` | `results/task2_results.csv`, `results/task2_embedding_2d.csv` |
| task3 | `scripts/task3_fewshot.py` | `results/task3_results.csv` (per episode), `results/task3_summary.csv`, predictions, confusion matrices |
| figures | `scripts/make_figures.py` | `results/figures/*.png` |

Runtime on a 10-core laptop CPU: about 3 min for the dataset stage, about 60 min for SSL training, and a few minutes for each task.

## Run an episode

```bash
python run_episode.py --episode path/to/episode [--target-room A|B] [--out preds.csv]
```

An episode folder contains `query/` (any mix of `.pcap`, 256-column `.csv`, or `(150, 256)` `.npy`). It can also contain `support/<ACTIVITY>/` with labelled target-room captures and `episode.yaml` (`target_room: A`).

Each query capture goes through the same decode → QC → window → normalise → route → embed → classify path as the development data. The output CSV has `window_id, room, label, confidence` (plus `room_confidence` and `source_file`). Each query window is saved as `window_XXXX.npy` next to it.

## Results snapshot (validation)

| Task | Result |
|---|---|
| 1. Room fingerprint | accuracy **1.000** on 274 fixed 150 × 256 windows (0.978 from a single observation) |
| 2. SSL linear probe, macro-F1 | Room A 0.932, Room B 0.992, joint 0.946 (random weights: 0.891 / 0.790 / 0.738; handcrafted: 0.932 / 0.992 / 0.909) |
| 3. Target A (B→A), macro-F1, K = 1/5/10/20 | primary 0.368 / 0.525 / 0.597 / 0.646; target-exclusive 0.395 / 0.570 / 0.711 / 0.793; shared 0.342 / 0.479 / 0.484 / 0.500 |
| 3. Target B (A→B), macro-F1, K = 1/5/10/20 | primary 0.338 / 0.596 / 0.647 / 0.707; target-exclusive 0.247 / 0.669 / 0.788 / 0.860; shared 0.400 / 0.546 / 0.553 / 0.605 |

Task 3 values are means over 5 support seeds. Shared SIT/STAND/WALK transferred directly from the source room stay at chance level (the Room B amplitudes fluctuate from row to row, and Room A activities are tied to locations). The main method therefore re-assigns them from clusters of the target room's unlabelled development windows, named by a motion prior (WALK > STAND > SIT), without any target shared labels. Full tables, ablations, and figures are in [report/report.md](report/report.md) and [results/](results/).

## Repository layout

```
configs/experiment.yaml      every tunable choice (QC threshold, windowing, normalisation, SSL, K, seeds)
src/csihar/
  pcap_reader.py             dpkt-based UDP payload extraction (port 5500), packet order preserved
  csi_decode.py              Nexmon header parsing + BCM4366c0 packed-float CSI reconstruction (own code)
  amplitude.py               |H| in FFT-bin order, common 256-column schema
  cleaning.py                QC: legacy-20 MHz frames, header consistency, zero/non-finite, duplicates
  ingest.py / splits.py      capture discovery, Room-B validation split (seed 42), intermediate CSVs, manifest
  windowing.py / dataset.py  deterministic 150 x 256 windows + provenance index
  normalization.py           per-packet gain, dev-fitted per-subcarrier standardisation, null-bin masking
  features.py                handcrafted baseline features, slow-time motion features
  models.py / ssl.py         temporal CNN encoder, SimCLR training with CSI augmentations
  room.py                    Task 1 router
  fewshot.py                 Task 3: alignment, prototype + logreg ensemble, Sinkhorn refinement, calibration, routing,
                             shared-activity relabelling, episodes
  inference.py               arbitrary-capture loading for run_episode.py
scripts/                     stage entry points (see table above)
run_all.py, run_episode.py   orchestration / episode entry point
report/report.md             technical report
results/                     metrics CSVs and figures
```

## Leakage controls (summary)

- The capture is the atomic unit everywhere: one capture never contributes windows to both splits, and K counts captures.
- The Room A validation set is the supplied `validation/` folder. The Room B validation set has 25 captures per class: candidate file names are sorted, then `random.Random(42).sample` is called with a fresh RNG per class.
- The normaliser, null-bin mask, router, SSL encoder, alignment statistics, and prototype refinement use development data only. Validation windows are only transformed and scored.
- Task 3 support captures come from the target room's development captures only. Shared-activity labels of the target room are never used.
- No metadata is used as a feature: the models see only the 150 × 256 amplitude array. Room A captures have fewer than 150 valid observations, so they are resampled to 150 rows rather than edge-padded; a padding block would encode capture length.
