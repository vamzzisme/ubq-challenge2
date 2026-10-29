# Cross-Room Few-Shot Adaptation for WiFi CSI Activity Recognition

CS60055 Ubiquitous Computing, Hackathon Challenge 2: technical report

*Team: <names / roll numbers>*. All reported numbers are **validation** results on the fixed validation sets defined by the challenge. They are not unseen-test accuracies.

---

## 1. Summary

We built a Python pipeline from the raw challenge package to 150 × 256 amplitude windows, then a router, a self-supervised encoder, and a few-shot cross-room adapter on top of it. Each item below is a validation result.

- **Preprocessing.**
  - We wrote our own BCM4366c0 packed-float CSI decoder for Room A.
  - We found that 71% of Room A frames are legacy 20 MHz frames that sound only 64 of the 256 subcarriers, including 47k frames that *do* carry sequence numbers. QC removes them on CSI content.
  - Both rooms are exported in a common 256-column CSV form with a full provenance trail (capture → intermediate CSV → window, plus config hash).
- **Task 1 (room fingerprint).** **100% validation accuracy** on 274 fixed 150-row windows. The accuracy is still 97.8% from a *single* observation and ≥ 99.6% from 5 observations. Room is close to trivially identifiable, and Section 7 explains why this should not be read as physical room recognition.
- **Task 2 (SSL).** Linear probes on the frozen SimCLR encoder reach macro-F1 **0.932** (Room A, 6 classes), **0.992** (Room B, 5 classes) and **0.946** (joint, 8 classes). The same architecture with random weights gives 0.891 / 0.790 / 0.738, and handcrafted features give 0.932 / 0.992 / 0.909.
- **Task 3 (cross-room few-label).** The main method is SSL embeddings, per-room CORAL alignment, a prototype + logistic-regression ensemble, Sinkhorn refinement and class-balance calibration on unlabelled target-development windows, and routing by the predicted room.
  - **Primary macro-F1 rises with K:** from 0.27 to 0.46 for target A (B→A) and from 0.21 to 0.43 for target B (A→B), for K = 1 → 20.
  - **Target-exclusive macro-F1 rises** from 0.40 to 0.79 (A) and from 0.25 to 0.86 (B).
  - **Shared SIT/STAND/WALK transferred without target labels stays at or below chance** (macro-F1 0.08–0.20). The errors are a systematic swap (WALK ↔ SIT), not a collapse onto one class.
  - **Environment-aware routing is the largest contributor:** removing it costs 0.14–0.16 macro-F1 at K = 20.
  - **SSL pretraining matters most for target B:** 0.43 vs 0.28 for a random encoder at K = 20.
  - **The ensemble and calibration** raise the average primary macro-F1 over all 8 (target, K) cells from 0.351 (prototype-only method) to 0.366, mainly through the target-exclusive classes.

## 2. Data inspection (Hint 1)

`scripts/inspect_data.py` writes `results/data_inspection.json`. The findings that shaped the pipeline are listed below.

**Room A (Experiment 2):**

- 1,084 pcaps (934 dev + 150 val, 25 per class). Link type is Ethernet; every packet is IPv4/UDP to port 5500.
- 234,021 packets in total (112–251 per capture). 234,019 payloads are 1,042 bytes, which is an 18-byte Nexmon header plus 256 × 4-byte words. Two payloads are malformed (726 bytes).
- Chip field 0x006a and a single chanspec 0xe02a (80 MHz). Every frame is core 0 / spatial stream 0.
- Each capture spans about 3.9 s, so the folder name `30ms` is the ping interval, not a duration.
- About half of all frames carry sequence number 0xFFFF. Decoding them shows energy only in the lowest 20 MHz sub-band: they are legacy 20 MHz frames, not 80 MHz observations.
- A further 47,306 frames *with* valid sequence numbers have the same legacy signature. After removing all legacy frames, the valid VHT80 frames sit in the last ~2 s of each capture at ≈27 Hz (≈36 ms spacing).
- 600 frames repeat a sequence number (MAC retransmissions).

**Room B (Experiment 3):**

- 1,201 CSVs, each exactly 500 rows × 256 integer columns, with no timestamps.
- 1,196 files use the header `Subcarrier_1..256`; 5 use `0..255`.
- 4 files do not follow the `*_Amp_*` naming (`Empty_11`, `Lying_0`, `Sitting_0`, `Standing_0`). They are unique captures (no duplicate content, checked by hash) and make up the documented total of 1,201. `model_plot_1.png` is ignored.
- Subcarriers are stored in FFT-bin order, the same as our Room A decoding: guard band at bins 123–133, VHT80 pilots at bins ±11/39/75/103.
- **Room B fluctuates far more from row to row than Room A.** Each row carries a strong ripple across subcarriers whose phase shifts from row to row (`fig_qc_examples.png`, right). As a result, the median correlation between consecutive rows over data subcarriers is 0.05–0.5, against 0.999 for decoded Room A frames, and a single subcarrier in an *empty* room jumps between ≈20 and ≈1,000. Only the window-level mean profile is stable. We tried alternative decodings of Room A packets (no autoscale, mantissa only, int16 pairs), and none reproduces this signature, so we treat it as a property of the supplied amplitudes. Section 7 discusses the consequences.
- **Room B rows are not interleaved streams.** The median correlation between row *i* and row *i + k* is flat for k = 1…6 (Empty ≈ 0.39, Lying ≈ 0.39, Sitting ≈ 0.07, Standing ≈ 0.08, Walking ≈ 0.12 at every lag; 60 sampled captures, data bins only). If the rows were two alternating antennas or transmitters, lag 2 would be close to 1 while lag 1 stayed low. Instead, each Room B row behaves like an independent snapshot around a stable mean, so row order carries no motion information in Room B.

## 3. Pipeline

```
Room A .pcap ──dpkt──▶ UDP payloads (port 5500, file order)
                         │  parse 18-byte Nexmon header (magic, RSSI, fc, MAC, seq, core/ss, chanspec, chip)
                         │  BCM4366c0 packed-float → complex CSI (own vectorised decoder)
                         ▼
                      |H|  (n_frames × 256, FFT-bin order)
                         │  QC: malformed, header-consistency, legacy-20 MHz, zero, non-finite,
                         │      duplicate sequence numbers, duplicate rows
Room B .csv ───────▶ 256 amplitude columns ── QC: non-finite, zero, negative, legacy-20 MHz, duplicate rows
                         ▼
        COMMON FORM: one CSV per capture, subcarrier_001..256, one row per valid observation
                     + manifest.csv (capture id, room, label, split, source, QC counts, content hash)
                         ▼
        WINDOWS: 150 × 256 float32 .npy  + index.csv (window → capture, rows, method, config hash)
                         ▼
        NORMALISATION (fitted on development only): per-packet RMS gain → per-subcarrier standardisation
                     → guard/DC bins zeroed
                         ▼
   ┌────────────── Task 1 router ───────────────┐   ┌──── Task 2 SSL encoder (no labels) ────┐
   │ time-mean + time-std profile → logistic reg │   │ temporal CNN, SimCLR on capture pairs │
   └───────────────────┬────────────────────────┘   └───────────────┬────────────────────────┘
                       ▼                                              ▼
      Task 3: routed room → per-room embedding alignment → prototype + logreg ensemble (source labels + K support)
              → Sinkhorn-balanced refinement + class-balance calibration on unlabelled target-dev windows
              → room-restricted label space
                         ▼
                  predictions: window_id, room, label, confidence
```

### 3.1 CSI reconstruction (Room A)

The 4366c0 does not export int16 I/Q. Each 32-bit word holds two 11-bit magnitudes with their signs (I: bits 18–28, sign bit 29; Q: bits 6–16, sign bit 17) and a shared 6-bit two's-complement exponent (bits 0–5). We reconstruct `sign·mantissa·2^exponent` with vectorised NumPy over all packets of a capture (`csi_decode.unpack_4366c0`).

Following the public Nexmon format description, each packet is then rescaled by a power of two so that its largest component occupies 10 bits. Components whose shifted exponent falls below −12 are flushed to zero. Amplitude is `|H|`, kept in FFT-bin order with no reordering, because Room B uses the same order.

The decoded spectra look physically right: zero DC, an 11-bin guard band in the centre, smooth frequency-selective fading, and pilots visible (`results/figures/fig_qc_examples.png`).

### 3.2 Quality control

| Rule | Room A rejected | Room B rejected | Why |
|---|---:|---:|---|
| malformed payload (magic/length) | 2 | n/a | not a 256-subcarrier record |
| header inconsistency | 0 | n/a | a different chip, chanspec, or chain is not the same observation type |
| legacy-20 MHz (max sub-band energy share > 0.9) | 165,209 | 44,425 | the frame only sounds 64 of the 256 subcarriers |
| zero / constant / non-finite / negative | 0 | 0 | zero-information or invalid CSI |
| duplicate sequence number | 600 | n/a | MAC retransmission of the same frame |
| duplicate consecutive row | 0 | 0 | repeated observation |
| capture with < 20 valid observations | 7 captures (6 dev, 1 val: `validation/sit/sit201`, 2 valid frames) | 0 | no meaningful window |

The legacy-frame statistic is cleanly bimodal on Room A: VHT80 frames have a maximum sub-band share ≤ 0.8 and legacy frames ≥ 0.95, so the 0.9 threshold is not sensitive. The rule works on CSI content rather than the sequence field, because 47,306 sequenced frames are also legacy. The same rule is applied to Room B so that both rooms pass an identical validity test; it removes 7.4% of Room B rows.

After QC, Room A has 68,210 valid observations (median 59 per capture) and Room B has 556,075 (308–500 per capture).

### 3.3 Splits

- **Room A validation:** the supplied `validation/` folder (149 usable captures).
- **Room B validation:** for each class in sorted label order, candidate file names are sorted lexicographically, then `random.Random(42).sample(candidates, 25)` is drawn with a fresh RNG per class. That gives 125 captures; the remaining 1,076 are development data.
- The capture is the atomic unit: every window carries its `capture_id`, and no capture appears in both splits.
- The manifest also stores an MD5 hash of each capture's cleaned content. Duplicated captures would be dropped, and none were found.

### 3.4 Windowing (150 × 256)

- **Validation captures:** exactly one window per capture, centred (`start = (n−150)//2`), so validation scoring is capture-level.
- **Development captures:** sliding windows from row 0 with stride 50. Room B gives about 7 windows per capture. These extra windows are training augmentation only; K still counts captures.
- **Short captures (all of Room A, 20–118 valid rows):** linear interpolation along the observation axis onto 150 evenly spaced points. Because the valid VHT80 frames are nearly uniformly spaced, this is a ≈2.5× uniform time stretch.

We prefer resampling to edge padding because a padding block's length is a direct function of capture length, which is a forbidden metadata cue for room or activity prediction. The interpolation is the documented price: Room A windows are smoother than raw observations.

Windows never mix captures. `index.csv` records the source file, intermediate CSV, rows used, method, and a hash of the QC and windowing configuration for every window.

### 3.5 Normalisation

1. **Per-packet gain:** each observation is divided by its RMS over data subcarriers. The 4366c0 export rescales each packet by a power of two, and AGC varies, so absolute packet level is not a physical quantity. This step is stateless.
2. **Per-subcarrier standardisation:** mean and standard deviation are fitted on **development windows only**, pooled over both rooms. Pooling means no room label is needed at transform time, which matters for the label-free SSL stage.
3. **Null bins:** bins whose development median is below 5% of the median level in either room (DC bins 0, 1, 255 and guard bins 123–133) are set to 0 after standardisation, so their noise is not amplified to unit variance. All 256 columns are kept.

Room-specific statistics are applied later, at the embedding level (Section 6), where the room comes from the router.

## 4. Task 1: recorded-environment identification

**Method.** The router is `csihar.room.RoomRouter`. For every subcarrier it takes the time-mean and time-std of the normalised window (512 features), then applies standardisation and L2-regularised logistic regression (C = 0.05). It is fitted on all 8,462 development windows with room labels and sees only the amplitude array.

| Representation (linear classifier) | Room accuracy, 274 validation windows |
|---|---:|
| profile router (headline, used for routing) | **1.000** (A: 149/149, B: 125/125) |
| SSL embedding probe | 1.000 |
| random-encoder embedding probe | 1.000 |
| handcrafted-feature probe | 1.000 |

**Accuracy vs window length.** For each length L, the router is trained on random L-row sub-windows of development windows and tested on the centred L-row sub-window of every fixed validation window (`results/figures/fig_room_vs_length.png`).

| L (observations) | 1 | 2 | 5 | 10 | 25 | 50 | 100 | 150 |
|---|---|---|---|---|---|---|---|---|
| accuracy | 0.978 | 1.000 | 0.996 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 |
| Room A windows | 0.960 | 1.000 | 0.993 | 0.993 | 1.000 | 1.000 | 1.000 | 1.000 |
| Room B windows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Interpretation.** The two recordings differ in many ways besides geometry:

- the static multipath profile (the intended signal);
- the guard-band noise floor;
- per-row variability (Section 2);
- smoothness (Room A windows are resampled).

Any of these separates the rooms from a handful of observations. The fingerprint therefore identifies the *recorded environment and session*, as the challenge text cautions. It does not show that a model could recognise a third room.

## 5. Task 2: learning without labels

**Encoder.** The encoder is a 1-D temporal CNN over the (T, 256) window: a 1 × 1 "spectral" convolution from 256 to 128 channels, then four temporal convolution blocks (kernel 7/5/5/3, strides 1/2/2/2). Mean and standard-deviation pooling over time feed a linear layer that produces a 128-d embedding. Pooling makes the encoder length-agnostic and captures both the static channel shape and the motion energy.

**SSL objective.** SimCLR NT-Xent (τ = 0.2) with a 2-layer projection head.

- Batches are built from *captures*, drawn uniformly so the two rooms are roughly balanced without using room labels. Capture identity is provenance, not a label.
- The two views of a capture come from two of its windows (different temporal positions for Room B).
- Augmentations: random crop of 96–150 rows, global gain ×[0.8, 1.25], smooth random per-subcarrier gain tilt, Gaussian noise, contiguous subcarrier-band masking, and time masking.
- Time reversal is deliberately excluded because SIT-DOWN and STAND-UP are near time reversals of each other.
- Training: 60 epochs, AdamW, one-cycle learning rate, on development windows of both rooms with no activity or room labels.

**Evaluation.** The encoder is frozen. A class-balanced logistic regression is fitted on labelled development embeddings and scored on validation windows. We compare against the same architecture with random weights and against handcrafted features: per-band temporal mean, std, mean |Δ| and range over 32 subcarrier bands; four Doppler-band energies; and the dominant-PC variance ratio.

The SSL loss decreased from 4.87 to 1.80 over 60 epochs (`fig_ssl_loss.png`). Linear-probe results on the validation windows (macro-F1 / balanced accuracy):

| Representation | Room A (6 cls) | Room B (5 cls) | Joint (8 cls) | Shared A→B | Shared B→A |
|---|---|---|---|---|---|
| **SSL encoder (frozen)** | **0.932** / 0.933 | **0.992** / 0.992 | **0.946** / 0.957 | 0.235 / 0.213 | 0.175 / 0.280 |
| same CNN, random weights | 0.891 / 0.892 | 0.790 / 0.792 | 0.738 / 0.771 | 0.167 / 0.333 | 0.168 / 0.333 |
| handcrafted features | 0.932 / 0.933 | 0.992 / 0.992 | 0.909 / 0.924 | 0.167 / 0.333 | 0.168 / 0.333 |

Self-supervision improves on the random-weight encoder in every within-room setting, and most on Room B (+0.20) and joint (+0.21). It matches or beats the handcrafted baseline. The "shared" columns train a probe on one room's SIT/STAND/WALK and test on the other room. They stay at chance for every representation, which is a first sign of the transfer problem in Task 3.

`fig_embedding_2d.png` shows UMAP plots of the validation embeddings. Rooms separate completely under both encoders. Within each room, the SSL embedding forms compact activity clusters (EMPTY, LYING, SIT, STANDUP), while the random encoder mixes them. This plot is qualitative.

## 6. Task 3: cross-room few-label adaptation

**Method.** One episode is defined by a target room T, K, and a support seed:

1. **Support.** For each T-exclusive activity, draw K distinct T-development captures (sorted capture ids, `numpy.default_rng(seed)`). All windows of a chosen capture are used. Support seeds 0–4 are run for every K, and we report mean ± std.
2. **Routing.** The Task 1 router predicts each validation window's room. The routed room selects (a) the embedding alignment statistics and (b) the label space: a window routed to room R can only receive R's activities.
3. **Environment-aware alignment.** SSL embeddings are whitened with the mean and (shrunk) full covariance of their room's *unlabelled* development embeddings: CORAL-style, a feature-level analogue of AdaBN. A diagonal variant is an ablation.
4. **Classifier.** An ensemble that averages the class probabilities of two heads trained on the same data:
   - a cosine nearest-prototype head (temperature 0.1), with per-capture means averaged so one capture cannot dominate a class;
   - a class-balanced logistic regression.

   Shared-activity classes are learned from all labelled source-development windows; T-exclusive classes come from the K support captures.
5. **Transductive refinement.** Soft k-means on the target room's unlabelled development windows (never validation), initialised from the prototypes. A Sinkhorn step balances class marginals, so the in-domain support prototypes cannot absorb every target window. Support means stay as anchors.
6. **Decision calibration.** Per-class logit offsets are fitted so that the predictions on the target room's unlabelled development windows are class-balanced (log-domain column scaling, as in Sinkhorn). The same offsets are then applied to validation windows. No labels and no validation data are used to fit them.

Ablations, all on validation:

- predicted-room routing vs oracle room vs no routing (pooled alignment, union label space);
- with vs without refinement, and with vs without calibration;
- the previous prototype-only method, and a logistic-regression-only head;
- random-weight encoder;
- handcrafted features;
- SSL embeddings of windows with their static per-subcarrier profile removed (to test the activity–location confound).

**Main method, mean ± std over 5 support seeds (validation).** Direction B→A means target room A; A→B means target room B.

| Metric | Target | K = 1 | K = 5 | K = 10 | K = 20 |
|---|---|---|---|---|---|
| primary macro-F1 | A | 0.266 ± 0.048 | 0.337 ± 0.021 | 0.397 ± 0.033 | 0.464 ± 0.026 |
| | B | 0.213 ± 0.044 | 0.379 ± 0.034 | 0.435 ± 0.029 | 0.434 ± 0.018 |
| target-exclusive macro-F1 | A (SIT-DOWN, STAND-UP, FALL) | 0.395 ± 0.068 | 0.570 ± 0.031 | 0.711 ± 0.029 | 0.793 ± 0.035 |
| | B (EMPTY, LYING) | 0.247 ± 0.045 | 0.669 ± 0.126 | 0.788 ± 0.069 | 0.860 ± 0.040 |
| shared macro-F1 (no target labels) | A | 0.138 ± 0.059 | 0.104 ± 0.020 | 0.083 ± 0.060 | 0.135 ± 0.042 |
| | B | 0.191 ± 0.076 | 0.185 ± 0.058 | 0.199 ± 0.075 | 0.151 ± 0.005 |
| balanced accuracy | A | 0.269 ± 0.054 | 0.336 ± 0.027 | 0.394 ± 0.039 | 0.467 ± 0.031 |
| | B | 0.227 ± 0.049 | 0.352 ± 0.042 | 0.402 ± 0.043 | 0.400 ± 0.025 |

The label-efficiency curve is `fig_label_efficiency.png`. Per-episode values are in `results/task3_results.csv`.

**Routing contribution** (primary macro-F1, `fig_routing.png`):

| Target | Routing | K = 1 | K = 5 | K = 10 | K = 20 |
|---|---|---|---|---|---|
| A | predicted room (router) | 0.266 | 0.337 | 0.397 | 0.464 |
| A | oracle room | 0.266 | 0.337 | 0.397 | 0.464 |
| A | no routing | 0.253 | 0.285 | 0.316 | 0.328 |
| B | predicted room (router) | 0.213 | 0.379 | 0.435 | 0.434 |
| B | oracle room | 0.213 | 0.379 | 0.435 | 0.434 |
| B | no routing | 0.135 | 0.237 | 0.277 | 0.272 |

Predicted routing is identical to oracle routing because the router is 100% accurate on validation. Removing routing means pooled alignment statistics and the union label space. With pooled statistics, source-room activities (for example EMPTY/LYING for target A) can steal target windows, and target embeddings are not re-centred. This costs up to 0.16 macro-F1.

**Ablations** (primary macro-F1, `fig_ablation.png`):

| Variant | A K=1 | A K=5 | A K=10 | A K=20 | B K=1 | B K=5 | B K=10 | B K=20 |
|---|---|---|---|---|---|---|---|---|
Unless the row says otherwise, every variant uses the main method's ensemble head, refinement and calibration.

| Variant | A K=1 | A K=5 | A K=10 | A K=20 | B K=1 | B K=5 | B K=10 | B K=20 |
|---|---|---|---|---|---|---|---|---|
| **main** (SSL, CORAL, ensemble, refine, calibrate, routed) | 0.266 | 0.337 | 0.397 | 0.464 | 0.213 | 0.379 | 0.435 | 0.434 |
| previous main (prototypes only, no calibration) | 0.258 | 0.300 | 0.376 | 0.447 | 0.235 | 0.378 | 0.395 | 0.422 |
| no calibration | 0.269 | 0.327 | 0.412 | 0.456 | 0.218 | 0.346 | 0.405 | 0.432 |
| no refinement | 0.239 | 0.389 | 0.474 | 0.494 | 0.171 | 0.307 | 0.398 | 0.425 |
| diagonal alignment instead of CORAL | 0.253 | 0.321 | 0.392 | 0.413 | 0.200 | 0.291 | 0.348 | 0.346 |
| logistic regression only (no refine, no calibration) | 0.249 | 0.385 | 0.469 | 0.484 | 0.147 | 0.325 | 0.384 | 0.417 |
| random-weight encoder | 0.260 | 0.359 | 0.370 | 0.408 | 0.182 | 0.244 | 0.289 | 0.282 |
| random-weight encoder, no refine | 0.269 | 0.375 | 0.431 | 0.454 | 0.183 | 0.235 | 0.285 | 0.284 |
| handcrafted | 0.296 | 0.347 | 0.422 | 0.506 | 0.178 | 0.290 | 0.349 | 0.378 |
| handcrafted, no refine | 0.254 | 0.335 | 0.396 | 0.443 | 0.229 | 0.301 | 0.364 | 0.414 |
| SSL, static profile removed | 0.215 | 0.275 | 0.338 | 0.402 | 0.189 | 0.310 | 0.359 | 0.378 |

What helps and what does not:

- **Routing is the clearest gain**, in both directions.
- **SSL pretraining** helps strongly when B is the target (0.38–0.43 vs 0.24–0.29 for a random encoder at K ≥ 5). For target A, SSL, random and handcrafted features are within noise of each other: Room A has only 928 development windows and its exclusive activities are short transitions. Handcrafted features are the best for target A, but the worst for target B among the non-random variants.
- **CORAL** beats diagonal alignment in all 8 cells.
- **Sinkhorn refinement** helps target B at every K (+0.01 to +0.07). It hurts target A at K ≥ 5, because it re-assigns ambiguous target windows to the (wrong) shared prototypes.
- **Ensemble and calibration.** Over all 8 cells, the ensemble alone (no calibration row) averages 0.358, against 0.351 for the previous prototype-only method. Calibration adds +0.03 for target B at K = 5 and 10, and lifts the average to 0.366.
  - Calibration does **not** fix the shared classes. Their errors are a label swap (target WALK lands on the SIT prototype and vice versa), not a collapse onto one class, and rebalancing the class frequencies cannot undo a swap.
  - The gain comes from the target-exclusive classes: exclusive macro-F1 rises from 0.513 to 0.570 (A, K = 5) and from 0.728 to 0.788 (B, K = 10).
- **Removing the static profile** (subtracting each window's per-subcarrier time mean before the SSL encoder) partly undoes the WALK swap: at K = 5, WALK recall is 0.20 (A) and 0.50 (B) instead of about 0. It loses more on every other class, so it is worse overall. This supports the activity–location confound explanation in Section 7.
- **Disclosure:** the main-method choices (CORAL, refinement, then the ensemble and calibration) were made by comparing variants on validation results, which the protocol allows for design decisions. In particular, the ensemble and calibration were adopted after a first 5-seed comparison against the prototype-only method on the same validation set.

**Confusion matrices at K = 5** (support seed 0; `fig_confusion_K5.png`):

| Target A | P | R | F1 |
|---|---|---|---|
| SIT | 0.24 | 0.29 | 0.26 |
| STAND | 0.10 | 0.08 | 0.09 |
| WALK | 0.00 | 0.00 | 0.00 |
| SIT-DOWN | 0.44 | 0.44 | 0.44 |
| STAND-UP | 0.73 | 0.76 | 0.75 |
| FALL | 0.56 | 0.60 | 0.58 |

| Target B | P | R | F1 |
|---|---|---|---|
| SIT | 0.15 | 0.16 | 0.15 |
| STAND | 0.41 | 0.56 | 0.47 |
| WALK | 0.00 | 0.00 | 0.00 |
| EMPTY | 1.00 | 0.84 | 0.91 |
| LYING | 0.79 | 0.44 | 0.56 |

In both directions, target WALK is mostly predicted as SIT (A: 18/25, B: 14/25), and target-B SIT is mostly predicted as WALK (16/25). Compared with the previous prototype-only method, F1 at seed 0 improves for every target-exclusive class (SIT-DOWN 0.33 → 0.44, STAND-UP 0.70 → 0.75, FALL 0.47 → 0.58, EMPTY 0.88 → 0.91, LYING 0.50 → 0.56). It also improves for target-B STAND (0.38 → 0.47) and target-A SIT (0.16 → 0.26). Target-B SIT drops (0.21 → 0.15).

## Figures

| | |
|---|---|
| ![QC](../results/figures/fig_qc_examples.png) | Decoded Room A spectrum (valid VHT80 frames vs rejected legacy-20 MHz frames), a Room A valid-frame spectrogram, and a raw Room B window. |
| ![label efficiency](../results/figures/fig_label_efficiency.png) | Label-efficiency curve: macro-F1 vs K per target room; target-labelled (exclusive) vs transferred shared activities. |
| ![embeddings](../results/figures/fig_embedding_2d.png) | Encoder diagnosis: UMAP of validation embeddings, coloured by room and by activity (SSL vs random weights). |
| ![room vs length](../results/figures/fig_room_vs_length.png) | Environment fingerprint: room accuracy vs window length (fixed validation capture pool). |
| ![routing](../results/figures/fig_routing.png) | Routing contribution: predicted-room routing vs oracle room vs no routing. |
| ![confusion](../results/figures/fig_confusion_K5.png) | Confusion matrices at K = 5 with per-activity precision, recall, and F1. |
| ![ablation](../results/figures/fig_ablation.png) | Ablation of representation, refinement, calibration, and head. |
| ![ssl loss](../results/figures/fig_ssl_loss.png) | SSL training loss. |

## 7. Discussion and known limitations

**Why shared activities do not transfer.** Within a room, activities are easy: probes reach 0.93–0.99 macro-F1. Across rooms, SIT/STAND/WALK transfer at or below chance under every representation and classifier we tried, and the errors are *systematic* (WALK ↔ SIT), not random. Three causes are visible in the data:

1. **The two archives are not the same measurement.**
   - Room A's decoded VHT80 frames are smooth in time (consecutive-row correlation 0.999). Motion appears as slow, coherent amplitude changes; WALK has 5–10× the temporal variation of SIT.
   - Room B's supplied amplitudes carry a strong frequency-domain ripple whose phase shifts from row to row (`fig_qc_examples.png`, right). Consecutive rows correlate at only 0.05–0.5, even in an EMPTY room.
   - In Room B the most "variable" classes are SIT and STAND, not WALK. After per-room alignment, "high temporal variation" therefore points to WALK in A but to SIT/STAND in B, which reproduces the observed swap.
   - We cannot recover phase or undo this from amplitude-only CSVs. Smoothing Room B in time did not make WALK stand out. The rows are also not interleaved streams that could be separated (Section 2): the row-to-row correlation is the same at every lag, so Room B has no temporal motion signal to align with Room A's.
   - Without target labels for SIT/STAND/WALK, nothing in the data identifies which target cluster corresponds to which source class. A label-free method can balance the predictions (our calibration step does), but it cannot resolve a swap.
2. **Activity–location confound.** Room A activities were recorded in different spatial zones. The static profile partly encodes *where* the person was, and that does not carry over to a room of different geometry.
3. **Session confound.** The rooms were recorded in separate sessions, so the rooms are trivially separable (Task 1). The alignment statistics remove first- and second-order room differences, but not class-conditional ones.

**What works.**

- Recognising the *target-exclusive* activities from very few captures: 0.67–0.79 macro-F1 at K = 5–10 for target B, and 0.71–0.79 at K = 10–20 for target A.
- Routing by environment.
- Self-supervised pretraining, which gives the best within-room representation and the best few-shot behaviour when Room B is the target.

**Limitations.**

- Room A windows are resampled (×2.5 time stretch) because no Room A capture has 150 valid 80 MHz observations.
- One Room A validation capture (`sit201`) has only 2 valid frames and is excluded, leaving 149 Room A validation windows.
- Results are capture-disjoint but *not* subject-disjoint.
- Only two environments are available.
- Validation results guided a small number of design choices (CORAL, refinement, the ensemble head, calibration), so they are not unseen-test estimates.
- Calibration assumes the target room's activities are roughly equally frequent. The challenge's validation sets are balanced (25 per class), but a real deployment with a very skewed activity mix would need a prior estimated in other ways.
- The random 80/20 split results in the original paper answer a different question and are not comparable.

**Future work.**

- Transfer through relational structure (for example matching the geometry of target clusters to source class prototypes) rather than absolute feature alignment.
- Alignment on the target-exclusive support classes.
- Recordings of shared activities with complex CSI in both rooms, so that phase-based features can be compared.

## 8. Pipeline independence, contribution, and AI-use disclosure

**Pipeline independence.** No source-author code, MATLAB, notebooks, processed `.mat` files, or result files were used or opened. The shared Drive folder also contains `Experiment-2.zip` and `Experiment-3.zip`, which bundle the authors' MATLAB code, results, and LSTM notebook. A first version of our download script fetched both. We listed their file names and extracted only the two ReadMe PDFs plus one data pcap (`validation/fall/fall120.pcap`) that had failed to download from the folder. The archives were then deleted without extracting or opening the code, results, or notebook. The download script now skips the archives.

The 4366c0 decoder follows the public Nexmon CSI format description and is our own vectorised implementation. Packet parsing below the UDP payload uses `dpkt`, a general-purpose PCAP library.

**Contribution statement.** *<to be completed by the team: who did ingestion/decoding, QC and windowing, SSL, few-shot adaptation, evaluation, and the report>*

**AI-use disclosure.** This project was developed with substantial help from an AI coding assistant (Claude, Anthropic). The assistant was used for data inspection, implementing the pipeline modules, running the experiments, and drafting this report. *<The team must review, understand and be able to explain every stage, and should edit this paragraph to describe their actual use, as required by the course AI-use rules.>*

## References

1. IEEE DataPort, CSI Human Activity dataset. https://ieee-dataport.org/open-access/csi-human-activity
2. J. Schäfer et al., "Human Activity Recognition Using CSI Information with Nexmon," *Applied Sciences* 11(19):8860, 2021.
3. F. Gringoli, M. Schulz, J. Link, M. Hollick, "Free Your CSI: A Channel State Information Extraction Platform for Modern Wi-Fi Chipsets," WiNTECH 2019 (Nexmon CSI).
4. T. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)," ICML 2020.
5. J. Snell, K. Swersky, R. Zemel, "Prototypical Networks for Few-shot Learning," NeurIPS 2017.
6. Y. Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation (AdaBN)," ICLR-W 2017.
7. B. Sun, K. Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation," ECCV-W 2016.
8. M. Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport," NeurIPS 2013; Y. Hu et al., "Leveraging the Feature Distribution in Transfer-based Few-Shot Learning (PT-MAP)," 2021.
