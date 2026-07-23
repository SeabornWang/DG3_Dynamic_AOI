# DG3 Trial-Window Dynamic-AOI Pipeline

This repository contains a research pipeline for processing DIKABLIS Glasses 3
(DG3) recordings in a marker-based human–robot interaction experiment. It
aligns an external experimental timeline with D-Lab gaze data and the DG3 scene
video, detects ArUco markers, constructs frame-level dynamic areas of interest
(AOIs), classifies gaze, and exports trial- and phase-level metrics.

The public version removes local machine paths and study-specific participant
ranges. Raw recordings and derived outputs are intentionally excluded.

## What the pipeline does

1. Reads a concatenated DG3 scene-camera video.
2. Reads a D-Lab gaze export while preserving recording time and UTC fields.
3. Aligns an external timeline using Unix-nanosecond timestamps.
4. Processes only padded trial windows instead of the full recording.
5. Detects ArUco markers and bridges short detection gaps with optical flow.
6. Builds dynamic AOIs for `Delivery`, `Task`, `RobotArm`, and `Gripper`.
7. Adds the optional `RobotNear` tolerance only to samples that would otherwise
   be classified as `Outside`.
8. Exports AOI availability, gaze-quality diagnostics, sample-level
   classifications, phase metrics, T2-minus-T1 deltas, and optional QA overlays.

## Repository layout

```text
.
├── data_raw/
│   └── P01/
│       ├── scene_video.mp4
│       ├── gaze_CsvData.txt
│       └── timeline.tsv
├── examples/
│   └── timeline_template.tsv
├── metadata/
│   └── marker_config_shared6.json
├── outputs/
├── scripts/
│   ├── hri_pipeline.py
│   ├── run_batch.ps1
│   └── run_participant.ps1
├── .gitignore
├── README.md
└── requirements.txt
```

Each participant folder should normally contain exactly one scene video, one
gaze export, and one timeline file. If a folder contains multiple candidate
files, pass their paths explicitly to `run_participant.ps1`.

## Requirements

- Python 3.10 or later
- Windows PowerShell 5.1 or PowerShell 7 for the wrapper scripts
- FFmpeg is optional but recommended for faster trial-clip export

Python packages are listed in `requirements.txt`. The OpenCV package must be
`opencv-contrib-python`, because the pipeline uses the `cv2.aruco` module.

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, the scripts can still use
the interpreter directly at `.venv\Scripts\python.exe`.

## Input files

### 1. Scene-camera video

Use one continuous scene-camera video per participant. If D-Lab stores the
recording as several source segments, concatenate them without changing their
order, frame rate, or duration.

Supported extensions in the PowerShell auto-detection workflow are `.mp4`,
`.avi`, `.mov`, and `.mkv`.

### 2. D-Lab gaze export

The parser supports UTF-8, UTF-8 BOM, UTF-16, and GB18030 text, with tab, comma,
or semicolon delimiters. It searches for common D-Lab column names, including:

- recording time: `rec_time`, `timestamp`, `time`, or `raw_time`;
- gaze X: the full D-Lab scene-camera gaze-X field, `Gaze X`, or `gaze_x`;
- gaze Y: the full D-Lab scene-camera gaze-Y field, `Gaze Y`, or `gaze_y`.

If auto-detection fails, pass `--time_col`, `--gaze_x_col`, and
`--gaze_y_col` to the Python entry point.

### 3. Experimental timeline

The preferred format is tab-separated `timeline.tsv`. CSV, TXT, XLSX, XLS, and
Apple Numbers files are also supported. The fine-phase schema requires:

```text
user_id
trial_index
task_id
t1_m_start_unix_ns
t1_m_end_unix_ns
t1_s_start_unix_ns
t1_s_end_unix_ns
t1_p_start_unix_ns
t1_p_end_unix_ns
t2_m_start_unix_ns
t2_m_end_unix_ns
t2_s_start_unix_ns
t2_s_end_unix_ns
t2_p_start_unix_ns
t2_p_end_unix_ns
```

Copy `examples/timeline_template.tsv` into a participant folder and add one row
per trial. Unix-nanosecond fields should remain full integer strings; avoid
spreadsheet scientific notation because it can lose timestamp precision.

An older coarse schema with explicit T0, T1, and T2 start/end fields remains
supported by the Python parser.

## Marker layout

The included configuration uses `DICT_4X4_50` and the following roles:

| Marker IDs | Role |
|---|---|
| 1, 2 | Delivery outer boundary |
| 3, 4 | Shared Delivery/Task boundary |
| 5, 6 | Task outer boundary |
| 9–12 | Robot arm |
| 13–15 | Gripper |

The table markers use a normalized shared coordinate system. Robot-arm markers
form a union of marker-centred AOIs. Gripper markers form an oriented or
connected ellipse depending on how many markers are visible.

Marker geometry and radius parameters are study-specific. Verify the resulting
AOIs on recorded videos before using the metrics in formal analysis.

## Run one participant

With the default folder structure:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_participant.ps1 `
  -ParticipantId P01
```

The script automatically uses the repository root, `.venv` when available,
`metadata\marker_config_shared6.json`, and `outputs\P01`.

Preview the resolved Python command without processing data:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_participant.ps1 `
  -ParticipantId P01 `
  -DryRun
```

When file detection is ambiguous:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_participant.ps1 `
  -ParticipantId P01 `
  -VideoPath "D:\research\P01\scene.mp4" `
  -GazePath "D:\research\P01\gaze_CsvData.txt" `
  -TimelineFile "D:\research\P01\timeline.tsv" `
  -OutputRoot "D:\research\derived"
```

Add `-NoOverlay` to skip QA overlay generation. Use
`-OverlayTrialIndices "2,5,7"` to render selected trials only.

## Run a participant range

Metrics only:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_batch.ps1 `
  -StartId 1 `
  -EndId 20 `
  -Stage Metrics `
  -MaxParallel 2
```

Metrics followed by QA overlays:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_batch.ps1 `
  -StartId 1 `
  -EndId 20 `
  -Stage All `
  -MaxParallel 2
```

The overlay stage renders all usable trials when mean AOI availability is below
the configured threshold. Otherwise, it renders one deterministic random trial
per participant for QA.

Do not set `MaxParallel` higher than the computer can support. Video decoding,
ArUco detection, and overlay writing are CPU-, memory-, and disk-intensive.

## Run Python directly

The Python entry point is cross-platform:

```bash
python scripts/hri_pipeline.py \
  --video data_raw/P01/scene_video.mp4 \
  --gaze data_raw/P01/gaze_CsvData.txt \
  --timeline_file data_raw/P01/timeline.tsv \
  --participant_id 1 \
  --config metadata/marker_config_shared6.json \
  --output_root outputs \
  --output_prefix P01
```

Use `python scripts/hri_pipeline.py --help` for the complete parameter list.

## Important outputs

For participant `P01`, the main outputs include:

| Output | Purpose |
|---|---|
| `P01_timeline_toi.csv` | Aligned trial and phase intervals |
| `P01_timeline_alignment_QA.csv` | Timestamp and video-coverage checks |
| `P01_windowed_marker_detection_rates.csv` | Marker/group detection rates |
| `P01_windowed_dynamic_aoi.csv` | Frame-level AOI geometry |
| `P01_trial_aoi_availability.csv` | AOI availability by trial |
| `P01_samples_with_aoi.csv` | Sample-level gaze labels and diagnostics |
| `P01_toi_metrics.csv` | Trial- and phase-level metrics |
| `P01_t2_minus_t1_deltas.csv` | Within-trial T2-minus-T1 differences |
| `P01_run_summary.csv` | Paths, settings, and run diagnostics |

Exact filenames can vary slightly by processing branch. The run summary is the
authoritative index for each completed run.

## Quality-control recommendations

- Inspect trial overlays before accepting AOI metrics.
- Report AOI availability together with dwell ratios.
- Treat `Outside` as a valid classification, not automatically as missing gaze.
- Distinguish missing/invalid gaze from valid gaze outside the predefined AOIs.
- Keep optical-flow bridging short; do not use stale AOIs across head turns.
- Verify scene resolution and the native D-Lab gaze coordinate resolution.
- Document any change to marker placement, radius, tolerance, or smoothing.
- Preserve the original D-Lab Study and scene-video segments separately.

## Data privacy and GitHub hygiene

Do not commit raw participant recordings, gaze samples, videos, logs, or derived
files containing absolute local paths. The included `.gitignore` excludes
`data_raw`, `outputs`, and common video formats, but review `git status` before
every public push.

The run-summary CSV stores input and output paths for reproducibility. Remove or
relativize those paths before sharing derived results publicly.

## Reproducibility notes

- The wrapper defaults reflect the original DG3 scene coordinate system
  (`2048 × 1152`) and the validated marker geometry used in this study.
- Different cameras, video exports, marker sizes, or room layouts require
  revalidation and possibly new parameter values.
- No participant data are included, so this repository provides the processing
  implementation rather than a fully executable example dataset.
- Add a release tag and record the Python/package versions used for any formal
  analysis.

## License and citation

No software license or citation metadata is included in this draft because
those choices should be made by the project owners before public release. Add
an explicit `LICENSE` and, if appropriate, a `CITATION.cff` file before making
the repository public.
