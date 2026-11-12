# DG3 Dynamic-AOI Pipeline for Human–Robot Interaction

A Python-based computer vision and multimodal data-processing pipeline for dynamic Area of Interest (AOI) construction and gaze analysis using DIKABLIS Glasses 3 (DG3) recordings.

The pipeline integrates experimental timeline synchronization, ArUco marker detection, optical-flow-based tracking, dynamic AOI construction, gaze classification, and trial-level behavioral and pupil metrics. It was developed for a marker-based human–robot interaction (HRI) study involving collaborative object delivery and placement.

**Repository:** [DG3_Dynamic_AOI](https://github.com/SeabornWang/DG3_Dynamic_AOI)

---

## 1. Overview

In collaborative human–robot interaction, participants' gaze targets may move continuously as the robot and task objects change position. Conventional static AOIs are therefore insufficient to characterize gaze allocation reliably throughout an interaction.

This project implements a dynamic AOI pipeline that uses visual markers attached to the experimental environment and robot components to reconstruct spatial regions frame by frame. It then maps gaze samples to these regions and derives behavioral measures for subsequent statistical analysis.

The pipeline is designed to:

- Synchronize experimental events, DG3 recording time, and scene-camera video time.
- Detect and track ArUco markers under partial occlusion and short detection gaps.
- Construct frame-level dynamic AOIs for task regions and robot components.
- Classify gaze samples according to their spatial relationship with the AOIs.
- Assess gaze-data quality and estimate temporal tracking loss.
- Extract trial- and phase-level gaze and pupil metrics.
- Generate visual overlays for quality inspection.

The public repository contains the processing implementation, marker configuration, and an example timeline template. Raw participant recordings and derived participant-level outputs are excluded.

## 2. Pipeline Architecture

The pipeline consists of five main processing stages.

```text
DG3 Scene-Camera Video
          +
D-Lab Gaze Export
          +
Experimental Timeline
          |
          v
1. Temporal Alignment
   - Recording-time parsing
   - UTC-to-video-time mapping
   - Trial and phase interval reconstruction
          |
          v
2. Windowed ArUco Detection
   - Trial-window video sampling
   - Marker detection
   - Short-gap optical-flow tracking
   - Scene-motion quality checks
          |
          v
3. Dynamic AOI Construction
   - Delivery and Task polygons
   - RobotArm geometry
   - Gripper geometry
   - AOI availability assessment
          |
          v
4. Gaze Classification & Quality Control
   - Gaze-to-AOI mapping
   - Invalid gaze identification
   - Tracking-loss estimation
   - Sample-level annotations
          |
          v
5. Behavioral Metrics & Visualization
   - AOI dwell duration and ratios
   - Gaze transition metrics
   - Pupil summaries
   - T1/T2 comparisons
   - Trial-level QA overlays
```

### Processing strategy

Instead of processing every frame in a full recording, the pipeline first aligns the experimental timeline and identifies trial-specific processing windows.

A configurable padding interval is added before and after each trial to preserve temporal context. By default, the pipeline processes an additional 1,000 ms before and 500 ms after each trial, while excluding this padding from the final trial metrics.

Marker tracking and AOI-related temporal states are initialized independently for each trial to prevent tracking information from carrying over between trials.

## 3. Key Technical Components

### 3.1 Temporal Alignment and Data Synchronization

The pipeline aligns three time sources:

- DG3 recording time (`rec_time`)
- UTC timestamps from the D-Lab export
- Unix-nanosecond timestamps from the external experimental timeline

The alignment procedure maps experimental event timestamps to DG3 video time and reconstructs trial-level and phase-level intervals.

Key features:

- Automatic detection of recording-time and gaze-coordinate columns.
- UTC-to-recording-time alignment.
- Configurable timeline clock offset and drift correction.
- Trial-window extraction to reduce unnecessary video processing.
- Alignment quality checks and video-coverage diagnostics.

### 3.2 ArUco Marker Detection and Tracking

The pipeline uses OpenCV's ArUco module with the `DICT_4X4_50` dictionary.

The marker configuration defines the spatial roles of the markers:

| Marker IDs | Role |
|---|---|
| 1, 2 | Delivery outer boundary |
| 3, 4 | Shared Delivery/Task boundary |
| 5, 6 | Task outer boundary |
| 9–12 | RobotArm |
| 13–15 | Gripper |

To improve robustness under partial occlusion, the pipeline combines direct ArUco detection with short-gap optical-flow tracking.

**Forward–Backward Lucas–Kanade Optical Flow**

The tracking procedure:

1. Tracks marker corners from the previous frame to the current frame.
2. Tracks the estimated corners backward to the previous frame.
3. Calculates forward–backward tracking error.
4. Rejects unreliable points and insufficient corner correspondences.
5. Estimates marker geometry and checks area changes, displacement, and frame boundaries.

Short optical-flow bridges are accepted only when the tracking quality satisfies configurable criteria.

**Global Scene-Motion Detection**

The pipeline also estimates global scene motion using distributed background features. Large or unreliable frame-to-frame camera motion triggers a tracking-state reset, preventing potentially stale marker estimates from being propagated through head turns or blurred transitions.

Direct ArUco detections remain usable during these resets.

A separate forward–backward validation procedure supports short table-marker holds when appropriate.

### 3.3 Dynamic AOI Construction

Four primary AOIs are reconstructed:

- `Delivery`
- `Task`
- `RobotArm`
- `Gripper`

**Delivery and Task**

The tabletop regions are reconstructed from their corresponding marker groups.

- With four usable reference markers, the pipeline constructs the region using downward boundary rays extended toward the bottom of the video frame.
- If the ray-based construction is unavailable, a validated quadrilateral fallback is used.
- With exactly three usable markers, a conservative direct triangle is constructed without inferring the missing marker.
- With fewer than three usable markers, the AOI is marked unavailable.

**RobotArm**

RobotArm markers (IDs 9–12) define the robot-arm interaction region.

- One visible marker: a marker-centred circular AOI.
- Two or more visible markers: a connected, rotated elliptical AOI enclosing the visible marker locations.

**Gripper**

Gripper markers (IDs 13–15) describe the end-effector region.

- One visible marker: an oriented elliptical AOI.
- Two or more visible markers: a connected elliptical AOI based on the visible marker positions.

AOI geometry is reconstructed over time, with configurable smoothing and tracking-quality checks.

**Optional RobotNear Classification**

An additional `RobotNear` label can capture gaze points located just outside the direct RobotArm or Gripper boundaries.

Importantly, this tolerance is applied only when a valid gaze point would otherwise be classified as `Outside`. It does not override direct classifications of `Task`, `Delivery`, `RobotArm`, or `Gripper`.

### 3.4 Gaze Classification and Data Quality Assessment

Each valid gaze sample is mapped to the corresponding frame-level AOI geometry.

The direct AOI classification priority is:

```text
Gripper > RobotArm > Delivery > Task
```

The pipeline distinguishes between:

- `Delivery`
- `Task`
- `RobotArm`
- `Gripper`
- `RobotNear`
- `Outside`
- `NoAOIData`
- `InvalidGaze`

`Outside` represents valid gaze located outside the predefined AOIs, whereas `NoAOIData` indicates that the corresponding AOI geometry is unavailable.

Invalid gaze is further characterized through row-level quality statuses, including:

- Missing time
- Missing gaze coordinates
- Non-numeric gaze coordinates
- Zero or negative gaze coordinates
- Coordinates outside the native recording frame
- Coordinates outside the AOI/video frame

**Temporal Gaze-Stream Quality**

D-Lab exports may contain interleaved rows with missing gaze coordinates. These rows are not automatically treated as tracking-loss events.

Instead, the pipeline estimates valid gaze coverage from temporal gaps between valid gaze samples, using a configurable continuity threshold.

This provides separate diagnostics for:

- Valid gaze coverage
- Estimated tracking-loss duration and ratio
- Longest gap between valid gaze samples
- Number and duration of estimated tracking-loss runs

Marker detection rates are calculated using all sampled video frames as the denominator, including frames in which no marker is detected.

### 3.5 Behavioral and Pupil Metrics

The pipeline exports sample-level classifications and computes overall, trial-level, and phase-level metrics.

**Gaze and AOI Metrics**

- AOI dwell duration
- AOI dwell ratios relative to total duration and valid gaze duration
- Gaze allocation across Delivery, Task, RobotArm, and Gripper
- RobotNear and Outside duration
- Gaze transitions and related behavioral summaries

**Pupil Metrics**

When corresponding pupil measurements are available in the D-Lab export:

- Mean pupil area
- Mean pupil height and width
- Equivalent pupil diameter derived from pupil area
- 95th-percentile pupil diameter
- Pupil diameter temporal slope
- Baseline-referenced pupil changes when a T0/baseline phase is available

**Trial and Phase Comparisons**

The pipeline supports phase-level summaries and within-trial comparisons, including:

- T1 and T2 metrics
- T2-minus-T1 differences
- Phase-specific gaze allocation and pupil summaries

Metrics are duration-weighted where appropriate, rather than relying solely on raw export-row counts.

### 3.6 Quality-Assurance Visualization

The pipeline can generate trial-level overlay videos containing:

- Original scene-camera footage
- Reconstructed dynamic AOI boundaries
- Gaze positions and classifications
- Trial-specific temporal information

Overlay generation is restricted to official trial intervals, excluding processing padding.

The overlays provide a visual means of checking marker tracking, AOI geometry, gaze classification, and potential failures before using the derived metrics in formal statistical analyses.

## 4. Code Architecture

The current implementation is organized into function-based processing stages within `scripts/hri_pipeline.py`.

The following functions provide useful entry points for inspecting the implementation:

| Function | Main responsibility |
|---|---|
| `read_table_auto()` | Robust parsing of D-Lab data exports |
| `resolve_alignment_from_gaze_file()` | Recording-time and UTC alignment |
| `read_timeline_file()` | Reading and validating experimental timelines |
| `convert_timeline_to_toi()` | Converting experimental timestamps into video-time intervals |
| `build_trial_processing_windows()` | Constructing padded trial-processing windows |
| `detect_markers_from_video_windows()` | Windowed ArUco detection and tracking |
| `track_marker_quad_lk()` | Forward–backward Lucas–Kanade marker tracking |
| `estimate_global_scene_motion()` | Detecting large or unreliable scene motion |
| `add_validated_short_table_holds()` | Validating short table-marker gaps |
| `build_dynamic_aoi_per_trial()` | Constructing AOIs independently for each trial |
| `build_dynamic_aoi()` | Frame-level AOI geometry reconstruction |
| `classify_one_gaze_detailed()` | Classifying an individual gaze sample |
| `classify_gaze()` | Gaze preprocessing, alignment, and AOI classification |
| `assign_gaze_quality_status()` | Identifying invalid gaze-data conditions |
| `estimate_gaze_stream_quality()` | Estimating temporal gaze coverage and tracking loss |
| `compute_metrics()` | Computing duration-weighted gaze and pupil metrics |
| `compute_toi_metrics()` | Computing trial- and phase-level metrics |
| `make_trial_overlay_videos()` | Generating trial-level QA visualizations |

The PowerShell scripts provide participant-level and batch-level execution interfaces.

## 5. Repository Structure

```text
DG3_Dynamic_AOI/
│
├── examples/
│   └── timeline_template.tsv
│
├── metadata/
│   └── marker_config_shared6.json
│
├── scripts/
│   ├── hri_pipeline.py
│   ├── run_batch.ps1
│   └── run_participant.ps1
│
├── requirements.txt
└── README.md
```

The following directories are expected in the local working environment but are not included in the public repository:

```text
data_raw/
└── P01/
    ├── scene_video.mp4
    ├── gaze_CsvData.txt
    └── timeline.tsv

outputs/
└── P01/
    └── ...
```

## 6. Requirements and Installation

### Requirements

- Python 3.10 or later
- OpenCV with the ArUco module (`opencv-contrib-python`)
- NumPy
- Pandas
- OpenPyXL
- xlrd
- numbers-parser (for Apple Numbers timeline files)

FFmpeg is optional for trial-clip export. When unavailable, the pipeline falls back to OpenCV-based frame extraction.

PowerShell 5.1 or PowerShell 7 is required for the provided PowerShell wrapper scripts. The Python entry point can be used independently on supported platforms.

### Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 7. Input Data

### 7.1 Scene-Camera Video

One continuous DG3 scene-camera video per participant is expected.

Supported video extensions in the PowerShell auto-detection workflow:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`

If D-Lab exports the recording as multiple video segments, concatenate them in their original order without changing the frame rate or temporal sequence.

### 7.2 D-Lab Gaze Export

The pipeline accepts TXT, CSV, and TSV gaze exports.

The parser supports common text encodings, including UTF-8, UTF-8 BOM, UTF-16, and GB18030, with automatic delimiter detection.

It attempts to identify the following fields:

| Variable | Candidate columns |
|---|---|
| Recording time | `rec_time`, `timestamp`, `time`, `raw_time` |
| Gaze X | D-Lab scene-camera gaze-X field, `Gaze X`, `gaze_x` |
| Gaze Y | D-Lab scene-camera gaze-Y field, `Gaze Y`, `gaze_y` |
| UTC | `UTC`, `utc`, `Utc` |

If automatic detection fails, specify the relevant columns using `--time_col`, `--gaze_x_col`, and `--gaze_y_col`.

### 7.3 Experimental Timeline

The preferred timeline format is tab-separated TSV.

The current fine-phase schema requires:

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

The phase abbreviations represent Motion (`m`), Safety Assessment (`s`), and Placement (`p`).

Copy `examples/timeline_template.tsv` into the corresponding participant folder and add one row per trial.

Supported timeline formats:

- TSV / TXT / CSV
- XLSX / XLS
- Apple Numbers (`.numbers`, backward compatibility)

An older coarse-phase schema containing explicit T0, T1, and T2 start/end timestamps is also supported.

**Important:** Unix-nanosecond timestamps should remain full integer values. Avoid spreadsheet scientific notation, which may introduce timestamp precision errors.

## 8. Running the Pipeline

### 8.1 Process One Participant

Using the PowerShell wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_participant.ps1 `
  -ParticipantId P01
```

The wrapper automatically resolves:

- Participant input directory
- Scene-camera video
- D-Lab gaze export
- Experimental timeline
- Marker configuration
- Python interpreter
- Output directory

The automatic input detection expects exactly one candidate scene video and one candidate gaze export in the participant folder.

To preview the resolved command without running the pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_participant.ps1 `
  -ParticipantId P01 `
  -DryRun
```

When automatic detection is ambiguous, specify the input paths explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_participant.ps1 `
  -ParticipantId P01 `
  -VideoPath "D:\research\P01\scene.mp4" `
  -GazePath "D:\research\P01\gaze_CsvData.txt" `
  -TimelineFile "D:\research\P01\timeline.tsv" `
  -OutputRoot "D:\research\derived"
```

To skip overlay generation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_participant.ps1 `
  -ParticipantId P01 `
  -NoOverlay
```

### 8.2 Batch Processing

Process participants P01–P20 and generate metrics:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_batch.ps1 `
  -StartId 1 `
  -EndId 20 `
  -Stage Metrics `
  -MaxParallel 2
```

Run both metrics and QA overlay stages:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_batch.ps1 `
  -StartId 1 `
  -EndId 20 `
  -Stage All `
  -MaxParallel 2
```

The batch runner supports three stages:

- `Metrics`: generate metrics and intermediate processing outputs.
- `Overlay`: generate QA overlays using existing metrics and marker results.
- `All`: execute both stages.

The overlay stage selects all usable trials when AOI availability is low. Otherwise, it selects one deterministic random usable trial per participant for visual inspection.

Parallel processing should be configured according to available CPU, memory, and disk resources.

### 8.3 Run Python Directly

The main Python entry point can also be executed without the PowerShell wrappers:

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

For the complete list of configurable parameters:

```bash
python scripts/hri_pipeline.py --help
```

Add `--make_overlay` to generate trial-level QA videos when running the Python entry point directly.

## 9. Main Outputs

All participant-specific outputs are saved under:

```text
outputs/P01/
```

The main output files include:

| Output | Description |
|---|---|
| `P01_timeline_toi.csv` | Aligned trial and phase intervals |
| `P01_timeline_alignment_QA.csv` | Timeline alignment and video-coverage checks |
| `P01_trial_processing_windows.csv` | Padded trial-processing intervals |
| `P01_windowed_marker_detections.csv` | Marker-level detection and tracking results |
| `P01_windowed_marker_frame_summary.csv` | Frame-level marker availability and tracking diagnostics |
| `P01_windowed_marker_detection_rates.csv` | Marker detection rates |
| `P01_windowed_dynamic_aoi.csv` | Frame-level dynamic AOI geometry |
| `P01_windowed_dynamic_aoi_summary.csv` | Frame-level AOI construction summaries |
| `P01_windowed_aoi_availability.csv` | Overall AOI availability |
| `P01_trial_aoi_availability.csv` | Trial-level AOI availability |
| `P01_samples_with_aoi.csv` | Sample-level gaze classifications and annotations |
| `P01_aoi_metrics_overall.csv` | Overall gaze and pupil metrics |
| `P01_aoi_label_summary.csv` | AOI label durations and ratios |
| `P01_gaze_export_row_quality.csv` | Row-level gaze export quality |
| `P01_gaze_stream_quality.csv` | Estimated gaze coverage and tracking-loss summary |
| `P01_gaze_tracking_loss_runs.csv` | Individual estimated tracking-loss intervals |
| `P01_time_alignment.csv` | Recording-time and video-time alignment information |
| `P01_diagnostics.csv` | Processing diagnostics |
| `P01_toi_metrics.csv` | Trial- and phase-level metrics |
| `P01_toi_label_summary.csv` | Trial- and phase-level AOI label summaries |
| `P01_t2_minus_t1_deltas.csv` | Within-trial T2-minus-T1 differences, when available |
| `P01_run_summary.csv` | Run configuration, paths, and output index |

Optional visual and video outputs:

```text
outputs/P01/
├── trial_overlays/
└── trial_clips/
```

The `trial_overlays` directory contains gaze and dynamic-AOI overlay videos when enabled. The `trial_clips` directory is generated only when raw trial-clip export is requested.

Exact intermediate files may vary depending on the selected processing options. The run-summary CSV provides the main index of the outputs generated in each run.

## 10. Quality Control and Analytical Considerations

The following checks are recommended before using the derived metrics in formal statistical analyses:

1. **Inspect AOI overlays:** Verify marker tracking, AOI boundaries, and gaze classification on representative trials.
2. **Report AOI availability:** Interpret dwell ratios together with the corresponding AOI availability rates.
3. **Distinguish gaze categories:** `Outside` represents valid gaze outside the defined AOIs and should not be treated as missing gaze.
4. **Check tracking loss:** Use temporal gaze-stream quality estimates rather than raw blank-row counts alone.
5. **Review optical-flow bridges:** Short-gap tracking should not be treated as equivalent to direct marker detection.
6. **Verify coordinate systems:** Confirm video resolution and native D-Lab gaze-coordinate resolution before processing.
7. **Review temporal alignment:** Inspect timeline alignment QA outputs and ensure all analyzed phases are covered by the video.
8. **Document configuration changes:** Record any changes to marker placement, AOI geometry, tracking thresholds, scaling, or smoothing parameters.

## 11. Reproducibility, Scope, and Limitations

This pipeline was developed for a specific marker-based HRI experimental setup and its associated DG3 scene-camera coordinate system.

Several parameters are therefore study-specific, including:

- ArUco marker roles and physical placement.
- Tabletop and robot-component AOI geometry.
- AOI expansion and tolerance distances.
- Gaze-coordinate scaling.
- Optical-flow tracking thresholds.
- Temporal smoothing and quality-control criteria.

Different camera resolutions, marker configurations, physical layouts, or recording settings require parameter adjustment and independent validation.

The current repository provides the processing implementation and configuration template rather than a complete sample dataset. Raw participant recordings and derived outputs are not distributed.

The pipeline is an offline, geometry- and computer-vision-based data-processing system. It does not implement a trained machine-learning model or claim generalization to arbitrary environments.

The main Python implementation is currently organized as a function-based script (`hri_pipeline.py`). The provided PowerShell wrappers support participant-level and batch-level execution.

A formal automated test suite is not currently included. Visual inspection of trial overlays and validation against the experimental setup are therefore important parts of the analysis workflow.

## 12. Data Privacy and Repository Hygiene

Participant recordings and gaze data may contain sensitive research information.

Do not commit:

- Raw scene-camera videos.
- Participant-level gaze exports.
- Experimental timelines containing identifiable information.
- Derived files containing sensitive participant data.
- Logs or output summaries containing private absolute paths.

The run-summary CSV records input and output paths for reproducibility. Sanitize or relativize these paths before sharing derived results publicly.

Before publishing updates, review the staged files and repository status to ensure that no restricted data or local machine information are included.

## 13. License and Citation

No software license or formal software citation metadata is currently included.

The appropriate license and citation information should be confirmed with the relevant project owners before redistribution or reuse.

For research use, please refer to the associated HRI study and its eventual publication when available.
