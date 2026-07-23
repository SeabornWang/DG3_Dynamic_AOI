"""
DG3 Trial-Window Dynamic-AOI Pipeline
=====================================

Purpose
-------
A standalone DG3 + ArUco pipeline for the revised pilot layout:

Upper row: 5 -- 3 -- 1
Lower row: 6 -- 4 -- 2

3, 4  : shared Delivery/Task boundary
5, 6  : Task outer boundary
1, 2  : Delivery outer boundary
9–12  : RobotArm markers; any one visible marker is sufficient
13–15  : Gripper markers; any one marker is sufficient

Key robustness updates
----------------------
* Delivery and Task are projected from a shared tabletop coordinate system.
  Any >=4 visible markers among IDs 1–6 use homography; any valid 3-marker
  non-collinear subset uses affine fallback. Therefore one temporarily hidden
  marker does not immediately invalidate an AOI.
* Short marker loss is bridged with Lucas–Kanade optical flow; stale AOIs are not frozen.
* Table AOIs are anchored by current direct ArUco detections.
* Table marker pairs are automatically oriented by image position, so IDs may be swapped within each row pair.
* Global scene-motion checks reset optical flow during head turns or blurred transitions.
* Table AOIs are rebuilt independently on every sampled frame from current usable markers; no lock, template calibration, or stale polygon hold.
* Equivalent pupil diameter in pixels is added to sample-, overall-, and TOI-level outputs.
* D-Lab interleaved blank rows receive zero analytical duration; AOI duration ratios use the valid gaze stream rather than raw export-row counts.
* RobotArm uses one circular AOI around every direct or optically tracked marker among IDs 9–12.
  The circles form a union, so all four markers do not need to appear together.
* Gripper uses one oriented ellipse for one visible marker and one connected ellipse for any 2–3 visible markers among IDs 13–15.
* Marker detection rates use ALL sampled frames as the denominator, including
  frames in which no marker is detected.
* Marker-only mode is available for field testing without gaze data.
* Full mode retains DG3 coordinate scaling, gaze classification, TOI metrics,
  T2-minus-T1 deltas, pupil summaries, UTC columns, and optional overlay video.
* D-Lab Triggered Tasks TXT exports can be read directly and converted into
  video-time, recording-time, and UTC/world-time TOI tables.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else Path.cwd()
DEFAULT_CONFIG = PROJECT_ROOT / "metadata" / "marker_config_shared6.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MAIN_AOIS = ["Delivery", "Task", "RobotArm", "Gripper"]
ROBOT_NEAR_LABEL = "RobotNear"
AVAILABILITY_AOIS = MAIN_AOIS + ["RobotTotal"]
AOI_PRIORITY = ["Gripper", "RobotArm", "Delivery", "Task"]
DEFAULT_TIME_CANDIDATES = ["rec_time", "timestamp", "time", "raw_time"]
DEFAULT_GAZE_X_CANDIDATES = [
    "Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze X",
    "Gaze X",
    "gaze_x",
]
DEFAULT_GAZE_Y_CANDIDATES = [
    "Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze Y",
    "Gaze Y",
    "gaze_y",
]

DEFAULT_MARKER_CONFIG: dict[str, Any] = {
    "dictionary": "DICT_4X4_50",
    "scheme": "shared6_pilot_simple_v6_35mm",
    "marker_roles": {
        "DeliveryOuter": [1, 2],
        "SharedBoundary": [3, 4],
        "TaskOuter": [5, 6],
        "RobotArm": [9, 10, 11, 12],
        "Gripper": [13, 14, 15],
    },
    "table_reference_points_normalized": {
        "1": [0.0, 0.0],
        "2": [1.0, 0.0],
        "3": [0.0, 0.5],
        "4": [1.0, 0.5],
        "5": [0.0, 1.0],
        "6": [1.0, 1.0],
    },
    "notes": {
        "9_12": "Attach to different visible faces/segments of the robot arm. Any one marker can define RobotArm AOI.",
        "13_15": "Place IDs 13, 14, and 15 on different gripper faces so at least one remains visible. Align each marker top edge with the gripper long axis when possible.",
    },
}


def require_runtime_dependencies() -> None:
    missing: list[str] = []
    if cv2 is None:
        missing.append("opencv-contrib-python")
    if np is None:
        missing.append("numpy")
    if pd is None:
        missing.append("pandas")

    if missing:
        raise SystemExit(
            "Missing required Python package(s): "
            + ", ".join(missing)
            + ". Install them with: python -m pip install -r requirements.txt"
        )


@dataclass
class PipelineConfig:
    video: Path
    output_prefix: str
    output_root: Path = DEFAULT_OUTPUT_ROOT
    gaze: Path | None = None
    toi_csv: Path | None = None
    triggered_tasks_txt: Path | None = None

    # Full-session timeline recorded on the experiment/robot computer.
    timeline_file: Path | None = None
    participant_id: int | None = None
    timeline_clock_offset_ms: float = 0.0
    timeline_clock_drift_ppm: float = 0.0
    export_trial_clips: bool = False
    clip_padding_before_ms: float = 0.0
    clip_padding_after_ms: float = 0.0

    # Efficient full-session processing:
    # align timeline first, then process only trial windows.
    trial_processing_padding_before_ms: float = 1000.0
    trial_processing_padding_after_ms: float = 500.0
    auto_find_triggered_tasks: bool = True
    video_start_rec_ms: float | None = None
    config_path: Path = DEFAULT_CONFIG
    sample_every_sec: float = 0.033
    max_nearest_ms: float = 80.0
    gaze_continuity_gap_ms: float = 100.0
    time_offset_ms: float = 0.0
    gaze_native_width: float = 2048.0
    gaze_native_height: float = 1152.0
    aoi_frame_width: float | None = None
    aoi_frame_height: float | None = None
    disable_scaling: bool = False
    delivery_scale: float = 1.05
    task_scale: float = 1.05

    # Stable-AOI policy: optical flow may bridge a short detection gap, but
    # pure hold-last geometry is disabled by default to prevent spatial drift.
    static_hold_last_ms: float = 0.0
    robot_arm_marker_radius_px: float = 110.0
    robot_hold_last_ms: float = 0.0
    gripper_base_radius_px: float = 70.0
    gripper_single_marker_radius_px: float = 125.0
    gripper_single_marker_major_radius_px: float = 150.0
    gripper_single_marker_minor_radius_px: float = 110.0
    gripper_single_marker_angle_offset_deg: float = 90.0

    # Table region construction:
    # with all four region markers visible, use the two vertical boundary
    # rays and extend them to the image bottom. With exactly three markers,
    # fall back to the conservative direct triangle.
    table_outer_expansion_px: float = 35.0
    table_ray_bottom_margin_px: float = 0.0

    # Robot-near tolerance is applied only after direct Gripper, RobotArm,
    # Delivery, and Task classification. Therefore it can only absorb gaze
    # that would otherwise be Outside.
    robot_near_enabled: bool = True
    robot_near_gripper_margin_px: float = 60.0
    robot_near_arm_margin_px: float = 40.0

    gripper_max_radius_px: float = 120.0
    gripper_hold_last_ms: float = 0.0

    detector_min_perimeter_rate: float = 0.012
    optical_flow_enabled: bool = True
    optical_flow_max_age_ms: float = 150.0
    table_optical_flow_max_age_ms: float = 250.0

    # Offline forward-backward validated table-marker hold.
    # A missing table marker is copied from its previous position only
    # when it reappears within a few sampled frames and the pre/post
    # displacement remains small. Otherwise the direct triangle is kept.
    table_validated_hold_enabled: bool = True
    table_validated_hold_max_frames: int = 3
    table_validated_hold_max_shift_px: float = 45.0
    table_ransac_reproj_threshold_px: float = 10.0
    table_reacquire_min_direct_markers: int = 3
    table_reacquire_consecutive_frames: int = 1
    table_corner_reacquire_consecutive_frames: int = 2
    table_corner_template_min_calibration_markers: int = 4
    table_corner_template_update_alpha: float = 0.15
    table_corner_min_pair_distance_norm: float = 0.45
    table_corner_ransac_reproj_threshold_px: float = 7.0
    table_corner_max_reprojection_error_px: float = 9.0
    table_min_direct_markers_while_locked: int = 2
    table_flow_only_bridge_max_age_ms: float = 120.0

    # Head-turn / scene-motion reset. Large camera motion or poor global
    # tracking invalidates optical-flow bridges until table markers are
    # directly reacquired.
    scene_motion_reset_enabled: bool = True
    scene_motion_max_features: int = 250
    scene_motion_min_valid_points: int = 24
    scene_motion_fb_error_px: float = 2.0
    scene_motion_median_threshold_px: float = 22.0
    scene_motion_p90_threshold_px: float = 55.0
    scene_motion_min_valid_fraction: float = 0.30
    optical_flow_fb_error_px: float = 1.5
    optical_flow_min_valid_corners: int = 3
    optical_flow_win_size_px: int = 31
    optical_flow_max_level: int = 3
    optical_flow_max_center_step_px: float = 140.0
    optical_flow_min_area_ratio: float = 0.45
    optical_flow_max_area_ratio: float = 2.20

    table_smoothing_alpha: float = 0.65
    robot_smoothing_alpha: float = 0.75
    gripper_smoothing_alpha: float = 0.75
    table_max_reprojection_error_px: float = 12.0
    table_max_centroid_jump_px: float = 260.0
    table_min_area_ratio: float = 0.40
    table_max_area_ratio: float = 2.50
    table_min_polygon_area_px2: float = 800.0
    temporal_quality_window_ms: float = 500.0
    make_overlay: bool = False
    overlay_start_ms: float | None = None
    overlay_end_ms: float | None = None
    overlay_fps: float = 10.0

    # Optional subset of Trial_Overall indices for QA overlay generation.
    # None means render every usable trial.
    overlay_trial_indices: tuple[int, ...] | None = None

    # Robust handling of recordings that start late, end early, or crash.
    # A phase is analyzed only when its complete explicit interval is covered
    # by the available video. No partial phase is clipped into an analysis
    # interval.
    require_full_phase_video_coverage: bool = True
    video_coverage_tolerance_ms: float = 250.0

    time_col: str | None = None
    gaze_x_col: str | None = None
    gaze_y_col: str | None = None
    marker_detections: Path | None = None
    marker_frame_summary: Path | None = None
    dynamic_aoi: Path | None = None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_table_auto(path: Path) -> pd.DataFrame:
    """
    Read DG3/D-Lab text exports robustly.

    Supports:
    - UTF-8 / UTF-8 BOM
    - UTF-16 LE/BE
    - GB18030 fallback
    - tab, comma, semicolon, and automatically detected delimiters

    The function rejects empty files and selects the parse result that
    contains the most plausible DG3 columns.
    """
    import io

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Gaze data file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Gaze data path is not a file: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(
            f"Gaze data file is empty (0 bytes): {path}\n"
            "Please export the DG3 *_CsvData.txt file again."
        )

    raw = path.read_bytes()
    if not raw.strip(b"\x00\r\n\t "):
        raise ValueError(
            f"Gaze data file contains no readable content: {path}"
        )

    encodings = [
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "gb18030",
        "latin1",
    ]

    suffix = path.suffix.lower()
    if suffix == ".txt":
        separators = ["\t", ",", ";", None]
    elif suffix == ".csv":
        separators = [",", "\t", ";", None]
    else:
        separators = [None, "\t", ",", ";"]

    parsed_candidates = []

    for encoding in encodings:
        try:
            decoded = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

        if not decoded.strip():
            continue

        for separator in separators:
            try:
                kwargs = {
                    "dtype": str,
                    "on_bad_lines": "skip",
                }

                if separator is None:
                    df = pd.read_csv(
                        io.StringIO(decoded),
                        sep=None,
                        engine="python",
                        **kwargs,
                    )
                else:
                    df = pd.read_csv(
                        io.StringIO(decoded),
                        sep=separator,
                        engine="python",
                        **kwargs,
                    )
            except Exception:
                continue

            if len(df.columns) == 0:
                continue

            df.columns = [
                str(column).replace("\ufeff", "").strip()
                for column in df.columns
            ]

            columns_lower = [column.lower() for column in df.columns]

            score = len(df.columns) * 5
            score += min(len(df), 1000) / 1000

            if any(column == "rec_time" for column in columns_lower):
                score += 100

            if any("gaze" in column and column.endswith("x") for column in columns_lower):
                score += 75

            if any("gaze" in column and column.endswith("y") for column in columns_lower):
                score += 75

            if any(column == "utc" for column in columns_lower):
                score += 20

            parsed_candidates.append(
                (score, encoding, separator, df)
            )

    if not parsed_candidates:
        preview = raw[:200]
        raise ValueError(
            f"Could not parse gaze data file: {path}\n"
            f"File size: {file_size} bytes\n"
            f"First bytes: {preview!r}"
        )

    parsed_candidates.sort(key=lambda item: item[0], reverse=True)
    _, selected_encoding, selected_separator, selected_df = parsed_candidates[0]

    separator_name = {
        "\t": "TAB",
        ",": "COMMA",
        ";": "SEMICOLON",
        None: "AUTO",
    }.get(selected_separator, repr(selected_separator))

    print("  Gaze file parser:")
    print(f"    path      = {path}")
    print(f"    size      = {file_size} bytes")
    print(f"    encoding  = {selected_encoding}")
    print(f"    delimiter = {separator_name}")
    print(
        f"    shape     = "
        f"{selected_df.shape[0]} rows x {selected_df.shape[1]} columns"
    )

    if selected_df.shape[1] <= 1:
        print(
            "    WARNING: only one column was parsed. "
            "Check whether the correct DG3 *_CsvData.txt file was selected."
        )

    return selected_df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_marker_config(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_MARKER_CONFIG, f, ensure_ascii=False, indent=2)
    return DEFAULT_MARKER_CONFIG


def get_aruco_dictionary(dict_name: str):
    mapping = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    }
    if dict_name not in mapping:
        raise ValueError(f"Unsupported ArUco dictionary: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(mapping[dict_name])


def role_for_marker(marker_id: int, config: dict[str, Any]) -> str:
    for role, ids in config["marker_roles"].items():
        if int(marker_id) in [int(x) for x in ids]:
            return role
    return "Unknown"


def detect_markers(frame: np.ndarray, aruco_dict, min_perimeter_rate: float) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = float(min_perimeter_rate)
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 10
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    rows: list[dict[str, Any]] = []
    if ids is None:
        return rows
    for marker_id, marker_corners in zip(ids.flatten(), corners):
        pts = marker_corners.reshape(4, 2)
        row: dict[str, Any] = {
            "marker_id": int(marker_id),
            "center_x": float(np.mean(pts[:, 0])),
            "center_y": float(np.mean(pts[:, 1])),
        }
        for i, (x, y) in enumerate(pts, start=1):
            row[f"corner_{i}_x"] = float(x)
            row[f"corner_{i}_y"] = float(y)
        rows.append(row)
    return rows



def corners_from_detection(detection: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [float(detection[f"corner_{i}_x"]), float(detection[f"corner_{i}_y"])]
            for i in range(1, 5)
        ],
        dtype=np.float32,
    )


def detection_from_corners(marker_id: int, corners: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(corners, dtype=float).reshape(4, 2)
    row: dict[str, Any] = {
        "marker_id": int(marker_id),
        "center_x": float(np.mean(pts[:, 0])),
        "center_y": float(np.mean(pts[:, 1])),
    }
    for i, (x, y) in enumerate(pts, start=1):
        row[f"corner_{i}_x"] = float(x)
        row[f"corner_{i}_y"] = float(y)
    return row


def marker_quad_area(corners: np.ndarray) -> float:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return abs(float(cv2.contourArea(pts)))


def marker_quad_is_valid(corners: np.ndarray, frame_width: int, frame_height: int) -> bool:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return False
    if marker_quad_area(pts) < 80.0:
        return False
    if not cv2.isContourConvex(pts.astype(np.float32)):
        return False
    margin = 40.0
    if (
        (pts[:, 0] < -margin).any()
        or (pts[:, 0] > frame_width + margin).any()
        or (pts[:, 1] < -margin).any()
        or (pts[:, 1] > frame_height + margin).any()
    ):
        return False
    return True


def track_marker_quad_lk(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_corners: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Track one marker quad with forward-backward Lucas–Kanade flow."""
    previous_pts = np.asarray(previous_corners, dtype=np.float32).reshape(-1, 1, 2)
    lk_params = dict(
        winSize=(int(cfg.optical_flow_win_size_px), int(cfg.optical_flow_win_size_px)),
        maxLevel=int(cfg.optical_flow_max_level),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    next_pts, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_pts,
        None,
        **lk_params,
    )
    if next_pts is None or status_forward is None:
        return None, {"flow_valid_corners": 0, "flow_fb_error_px": np.nan}

    back_pts, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        next_pts,
        None,
        **lk_params,
    )
    if back_pts is None or status_backward is None:
        return None, {"flow_valid_corners": 0, "flow_fb_error_px": np.nan}

    fb_error = np.linalg.norm(
        previous_pts.reshape(-1, 2) - back_pts.reshape(-1, 2),
        axis=1,
    )
    valid = (
        status_forward.reshape(-1).astype(bool)
        & status_backward.reshape(-1).astype(bool)
        & np.isfinite(fb_error)
        & (fb_error <= float(cfg.optical_flow_fb_error_px))
    )
    valid_count = int(valid.sum())
    diagnostics = {
        "flow_valid_corners": valid_count,
        "flow_fb_error_px": float(np.median(fb_error[valid])) if valid_count else np.nan,
    }
    if valid_count < int(cfg.optical_flow_min_valid_corners):
        return None, diagnostics

    old_valid = previous_pts.reshape(-1, 2)[valid]
    new_valid = next_pts.reshape(-1, 2)[valid]
    affine, _ = cv2.estimateAffinePartial2D(
        old_valid,
        new_valid,
        method=cv2.LMEDS,
    )
    if affine is None or not np.isfinite(affine).all():
        return None, diagnostics

    old_all_h = np.column_stack(
        [previous_pts.reshape(-1, 2), np.ones(4, dtype=np.float32)]
    )
    tracked = (affine @ old_all_h.T).T.astype(np.float32)

    old_area = marker_quad_area(previous_corners)
    new_area = marker_quad_area(tracked)
    area_ratio = new_area / old_area if old_area > 0 else np.nan
    center_step = float(
        np.linalg.norm(
            tracked.mean(axis=0)
            - np.asarray(previous_corners, dtype=float).reshape(4, 2).mean(axis=0)
        )
    )
    diagnostics["flow_area_ratio"] = area_ratio
    diagnostics["flow_center_step_px"] = center_step

    frame_height, frame_width = current_gray.shape[:2]
    if not marker_quad_is_valid(tracked, frame_width, frame_height):
        return None, diagnostics
    if not np.isfinite(area_ratio) or not (
        float(cfg.optical_flow_min_area_ratio)
        <= area_ratio
        <= float(cfg.optical_flow_max_area_ratio)
    ):
        return None, diagnostics
    if center_step > float(cfg.optical_flow_max_center_step_px):
        return None, diagnostics

    return tracked, diagnostics



def estimate_global_scene_motion(
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """
    Estimate frame-to-frame camera motion using distributed background features.

    This is not used to construct AOIs. It is only a safety gate: when camera
    motion is too large or the frame transition is too blurred/unreliable,
    optical-flow marker bridges are reset. Direct ArUco detections remain valid.
    """
    result: dict[str, Any] = {
        "scene_motion_feature_count": 0,
        "scene_motion_valid_count": 0,
        "scene_motion_valid_fraction": np.nan,
        "scene_motion_median_px": np.nan,
        "scene_motion_p90_px": np.nan,
        "scene_motion_large": False,
        "scene_motion_unreliable": False,
        "scene_motion_reset": False,
    }
    if previous_gray is None or not cfg.scene_motion_reset_enabled:
        return result

    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=int(cfg.scene_motion_max_features),
        qualityLevel=0.01,
        minDistance=12,
        blockSize=7,
        useHarrisDetector=False,
    )
    if points is None or len(points) == 0:
        result["scene_motion_unreliable"] = True
        result["scene_motion_reset"] = True
        return result

    result["scene_motion_feature_count"] = int(len(points))
    lk_params = dict(
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        **lk_params,
    )
    if next_points is None or forward_status is None:
        result["scene_motion_unreliable"] = True
        result["scene_motion_reset"] = True
        return result

    back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        next_points,
        None,
        **lk_params,
    )
    if back_points is None or backward_status is None:
        result["scene_motion_unreliable"] = True
        result["scene_motion_reset"] = True
        return result

    old_xy = points.reshape(-1, 2)
    new_xy = next_points.reshape(-1, 2)
    back_xy = back_points.reshape(-1, 2)
    fb_error = np.linalg.norm(old_xy - back_xy, axis=1)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(fb_error)
        & (fb_error <= float(cfg.scene_motion_fb_error_px))
    )
    valid_count = int(valid.sum())
    feature_count = int(len(points))
    valid_fraction = valid_count / feature_count if feature_count else 0.0

    result["scene_motion_valid_count"] = valid_count
    result["scene_motion_valid_fraction"] = valid_fraction

    if valid_count:
        displacement = np.linalg.norm(new_xy[valid] - old_xy[valid], axis=1)
        median_px = float(np.median(displacement))
        p90_px = float(np.percentile(displacement, 90))
        result["scene_motion_median_px"] = median_px
        result["scene_motion_p90_px"] = p90_px
        result["scene_motion_large"] = bool(
            median_px > float(cfg.scene_motion_median_threshold_px)
            or p90_px > float(cfg.scene_motion_p90_threshold_px)
        )

    result["scene_motion_unreliable"] = bool(
        valid_count < int(cfg.scene_motion_min_valid_points)
        or valid_fraction < float(cfg.scene_motion_min_valid_fraction)
    )
    result["scene_motion_reset"] = bool(
        result["scene_motion_large"] or result["scene_motion_unreliable"]
    )
    return result


def get_video_properties(video_path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    props = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": float(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if props["fps"] <= 0:
        raise ValueError("Video FPS could not be read.")
    return props


def longest_false_run_ms(flags: list[bool], timestamps: list[int]) -> float:
    if not flags or not timestamps:
        return 0.0
    if len(timestamps) > 1:
        dt = float(np.median(np.diff(np.asarray(timestamps, dtype=float))))
    else:
        dt = 0.0
    max_len = 0
    current = 0
    for flag in flags:
        if flag:
            current = 0
        else:
            current += 1
            max_len = max(max_len, current)
    return max_len * dt



def resolve_alignment_from_gaze_file(
    gaze_path: Path,
    video_props: dict[str, float],
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """
    Read only the timing columns needed to align timeline UTC to DG3/video time.

    This runs before marker detection, allowing the video to be processed only
    inside the ten trial windows.
    """
    gaze_df = read_table_auto(gaze_path)
    time_col = choose_column(
        gaze_df,
        cfg.time_col,
        DEFAULT_TIME_CANDIDATES,
        "time",
    )
    utc_col = next(
        (column for column in ["UTC", "utc", "Utc"] if column in gaze_df.columns),
        None,
    )
    if utc_col is None:
        raise ValueError(
            "The gaze file has no UTC column. The timeline recorded on another "
            "computer cannot be aligned without UTC or an explicit converted TOI."
        )

    rec_time_series = gaze_df[time_col].apply(parse_time_to_ms)
    utc_numeric_series = pd.to_numeric(
        gaze_df[utc_col],
        errors="coerce",
    )
    alignment = resolve_time_alignment(
        rec_time_ms=rec_time_series,
        utc_ms_raw=utc_numeric_series,
        video_props=video_props,
        cfg=cfg,
        triggered_tasks_raw=None,
    )
    alignment["gaze_time_column"] = time_col
    alignment["gaze_utc_column"] = utc_col

    print("\n[0/5] Resolve DG3 UTC / recording / video alignment")
    print(f"  gaze time column          = {time_col}")
    print(f"  gaze UTC column           = {utc_col}")
    print(
        f"  recording origin UTC ms   = "
        f"{alignment['recording_origin_utc_ms']:.3f}"
    )
    print(
        f"  video start rec ms        = "
        f"{alignment['video_start_rec_ms']:.3f}"
    )
    print(
        f"  alignment method          = "
        f"{alignment['video_start_alignment_method']}"
    )
    return alignment


def build_trial_processing_windows(
    toi: pd.DataFrame,
    video_props: dict[str, float],
    cfg: PipelineConfig,
    out_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    """
    Build one independent processing window per Trial_Overall interval.

    Padding frames are used only for marker/AOI initialization. Metrics remain
    restricted to the original T0/T1/T2 boundaries.

    Adjacent padded windows are clipped at the midpoint of the inter-trial gap,
    preventing duplicated video frames and ensuring that temporal states reset
    between trials.
    """
    overall = toi[
        toi["phase"].astype(str).eq("Trial_Overall")
    ].copy()
    if overall.empty:
        raise ValueError("No Trial_Overall rows were found in the timeline TOI.")

    overall = overall.sort_values(
        ["trial_index", "start_video_ms"]
    ).reset_index(drop=True)

    video_duration_ms = (
        float(video_props["frame_count"])
        / float(video_props["fps"])
        * 1000.0
    )

    windows: list[dict[str, Any]] = []
    for _, row in overall.iterrows():
        analysis_start = float(row["start_video_ms"])
        analysis_end = float(row["end_video_ms"])
        windows.append(
            {
                "processing_window_id": int(row["trial_index"]),
                "trial_id": str(row["trial_id"]),
                "trial_index": int(row["trial_index"]),
                "task_id": int(row["task_id"]),
                "analysis_start_video_ms": analysis_start,
                "analysis_end_video_ms": analysis_end,
                "processing_start_video_ms": max(
                    0.0,
                    analysis_start
                    - float(cfg.trial_processing_padding_before_ms),
                ),
                "processing_end_video_ms": min(
                    video_duration_ms,
                    analysis_end
                    + float(cfg.trial_processing_padding_after_ms),
                ),
                "padding_before_ms": float(
                    cfg.trial_processing_padding_before_ms
                ),
                "padding_after_ms": float(
                    cfg.trial_processing_padding_after_ms
                ),
            }
        )

    # Remove overlap between adjacent padded windows.
    for index in range(len(windows) - 1):
        current = windows[index]
        following = windows[index + 1]
        if (
            current["processing_end_video_ms"]
            > following["processing_start_video_ms"]
        ):
            midpoint = (
                current["analysis_end_video_ms"]
                + following["analysis_start_video_ms"]
            ) / 2.0
            current["processing_end_video_ms"] = max(
                current["analysis_end_video_ms"],
                min(current["processing_end_video_ms"], midpoint),
            )
            following["processing_start_video_ms"] = min(
                following["analysis_start_video_ms"],
                max(following["processing_start_video_ms"], midpoint),
            )

    windows_df = pd.DataFrame(windows)
    windows_df["processing_duration_ms"] = (
        windows_df["processing_end_video_ms"]
        - windows_df["processing_start_video_ms"]
    )
    windows_df["analysis_duration_ms"] = (
        windows_df["analysis_end_video_ms"]
        - windows_df["analysis_start_video_ms"]
    )

    path = out_dir / f"{cfg.output_prefix}_trial_processing_windows.csv"
    write_csv(windows_df, path)

    full_duration = video_duration_ms
    processed_duration = float(
        windows_df["processing_duration_ms"].sum()
    )
    reduction = (
        1.0 - processed_duration / full_duration
        if full_duration > 0
        else np.nan
    )

    print("\n  Trial processing windows:")
    print(
        windows_df[
            [
                "trial_index",
                "task_id",
                "processing_start_video_ms",
                "processing_end_video_ms",
                "processing_duration_ms",
            ]
        ].to_string(index=False)
    )
    print(
        f"  processed video duration  = {processed_duration / 1000.0:.1f} s"
    )
    print(
        f"  full video duration       = {full_duration / 1000.0:.1f} s"
    )
    print(
        f"  expected frame reduction  = {reduction:.1%}"
    )
    print(f"  saved: {path}")

    return windows_df, path


def _marker_detection_rate_tables(
    detections_df: pd.DataFrame,
    frame_summary_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    total = len(frame_summary_df)
    rate_rows: list[dict[str, Any]] = []

    for marker_id in range(1, 16):
        usable_col = f"marker_{marker_id}_detected"
        direct_col = f"marker_{marker_id}_direct"
        flow_col = f"marker_{marker_id}_flow"

        usable_count = (
            int(frame_summary_df[usable_col].sum())
            if total and usable_col in frame_summary_df.columns
            else 0
        )
        direct_count = (
            int(frame_summary_df[direct_col].sum())
            if total and direct_col in frame_summary_df.columns
            else 0
        )
        flow_count = (
            int(frame_summary_df[flow_col].sum())
            if total and flow_col in frame_summary_df.columns
            else 0
        )

        rate_rows.append(
            {
                "metric_type": "marker",
                "metric": f"marker_{marker_id}",
                "detected_frames": usable_count,
                "direct_frames": direct_count,
                "flow_bridged_frames": flow_count,
                "sampled_frames": total,
                "rate": usable_count / total if total else np.nan,
                "direct_rate": direct_count / total if total else np.nan,
                "flow_bridge_rate": flow_count / total if total else np.nan,
                "role": role_for_marker(marker_id, config),
            }
        )

    group_cols = {
        "table_reference_at_least_2": "At least two table markers usable",
        "table_reference_at_least_3": "At least three table markers usable",
        "table_reference_at_least_4": "At least four table markers usable",
        "shared_boundary_complete": "Both shared-boundary markers usable",
        "robot_arm_any": "At least one RobotArm marker usable",
        "gripper_any": "At least one Gripper marker usable",
        "gripper_multiple": "At least two Gripper markers usable",
    }

    for column, description in group_cols.items():
        count = (
            int(frame_summary_df[column].sum())
            if total and column in frame_summary_df.columns
            else 0
        )
        rate_rows.append(
            {
                "metric_type": "group",
                "metric": column,
                "detected_frames": count,
                "direct_frames": np.nan,
                "flow_bridged_frames": np.nan,
                "sampled_frames": total,
                "rate": count / total if total else np.nan,
                "direct_rate": np.nan,
                "flow_bridge_rate": np.nan,
                "role": description,
            }
        )

    return pd.DataFrame(rate_rows)


def detect_markers_from_video_windows(
    video_path: Path,
    config: dict[str, Any],
    out_dir: Path,
    cfg: PipelineConfig,
    processing_windows: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Path,
    Path,
    Path,
    dict[str, float],
]:
    """
    Detect markers only within the padded trial windows.

    Optical-flow state is reset at the beginning of every trial.
    Timestamps and frame indices remain global to the original full video.
    """
    props = get_video_properties(video_path)
    fps = float(props["fps"])
    step_frames = max(
        1,
        int(round(fps * float(cfg.sample_every_sec))),
    )
    aruco_dict = get_aruco_dictionary(config["dictionary"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    detection_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    sampled = 0

    print("\n[1/5] Windowed marker detection")
    print(f"  video: {video_path}")
    print(
        f"  FPS={fps:.3f}, frames={int(props['frame_count'])}, "
        f"sample every {step_frames} frame(s)"
    )
    print(f"  independent trial windows: {len(processing_windows)}")

    for _, window in processing_windows.sort_values(
        "trial_index"
    ).iterrows():
        trial_id = str(window["trial_id"])
        trial_index = int(window["trial_index"])
        task_id = int(window["task_id"])
        window_id = int(window["processing_window_id"])

        start_frame = max(
            0,
            int(
                np.floor(
                    float(window["processing_start_video_ms"])
                    / 1000.0
                    * fps
                )
            ),
        )
        end_frame = min(
            int(props["frame_count"]) - 1,
            int(
                np.ceil(
                    float(window["processing_end_video_ms"])
                    / 1000.0
                    * fps
                )
            ),
        )

        print(
            f"  Trial {trial_index:02d} / Task {task_id:02d}: "
            f"frames {start_frame}–{end_frame}"
        )

        # Critical: reset temporal state at every trial boundary.
        previous_sample_gray: np.ndarray | None = None
        tracked_states: dict[int, dict[str, Any]] = {}

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame

        while frame_index <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            relative_frame = frame_index - start_frame
            if relative_frame % step_frames != 0:
                frame_index += 1
                continue

            ts = int(round(frame_index / fps * 1000.0))
            current_gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            scene_motion = estimate_global_scene_motion(
                previous_sample_gray,
                current_gray,
                cfg,
            )
            if bool(scene_motion.get("scene_motion_reset", False)):
                tracked_states = {}

            direct_detections = detect_markers(
                frame,
                aruco_dict,
                cfg.detector_min_perimeter_rate,
            )
            direct_by_id = {
                int(detection["marker_id"]): detection
                for detection in direct_detections
            }

            combined_detections: list[dict[str, Any]] = []
            current_states: dict[int, dict[str, Any]] = {}

            for marker_id, detection in direct_by_id.items():
                corners = corners_from_detection(detection)
                detection = dict(detection)
                detection.update(
                    {
                        "detection_source": "aruco_direct",
                        "tracking_age_ms": 0.0,
                        "flow_valid_corners": 4,
                        "flow_fb_error_px": 0.0,
                        "flow_area_ratio": 1.0,
                        "flow_center_step_px": 0.0,
                    }
                )
                combined_detections.append(detection)
                current_states[marker_id] = {
                    "corners": corners,
                    "last_direct_ts": ts,
                }

            if (
                cfg.optical_flow_enabled
                and previous_sample_gray is not None
                and tracked_states
                and not bool(
                    scene_motion.get("scene_motion_reset", False)
                )
            ):
                for marker_id, state in tracked_states.items():
                    if marker_id in direct_by_id:
                        continue

                    tracking_age_ms = float(
                        ts - int(state["last_direct_ts"])
                    )
                    max_tracking_age_ms = (
                        float(cfg.table_optical_flow_max_age_ms)
                        if int(marker_id) in {1, 2, 3, 4, 5, 6}
                        else float(cfg.optical_flow_max_age_ms)
                    )
                    if tracking_age_ms > max_tracking_age_ms:
                        continue

                    tracked_corners, flow_diag = track_marker_quad_lk(
                        previous_sample_gray,
                        current_gray,
                        np.asarray(
                            state["corners"],
                            dtype=np.float32,
                        ),
                        cfg,
                    )
                    if tracked_corners is None:
                        continue

                    detection = detection_from_corners(
                        marker_id,
                        tracked_corners,
                    )
                    detection.update(
                        {
                            "detection_source": "optical_flow",
                            "tracking_age_ms": tracking_age_ms,
                            **flow_diag,
                        }
                    )
                    combined_detections.append(detection)
                    current_states[marker_id] = {
                        "corners": tracked_corners,
                        "last_direct_ts": int(
                            state["last_direct_ts"]
                        ),
                    }

            tracked_states = current_states
            previous_sample_gray = current_gray

            usable_ids = sorted(
                {
                    int(detection["marker_id"])
                    for detection in combined_detections
                }
            )
            direct_ids = sorted(direct_by_id)
            flow_ids = sorted(
                set(usable_ids) - set(direct_ids)
            )

            common_metadata = {
                "processing_window_id": window_id,
                "trial_id": trial_id,
                "trial_index": trial_index,
                "task_id": task_id,
            }

            for detection in combined_detections:
                detection.update(
                    {
                        "source": video_path.name,
                        "frame_index": frame_index,
                        "timestamp_ms": ts,
                        "role": role_for_marker(
                            int(detection["marker_id"]),
                            config,
                        ),
                        **common_metadata,
                    }
                )
                detection_rows.append(detection)

            usable_set = set(usable_ids)
            table_count = len(
                usable_set.intersection({1, 2, 3, 4, 5, 6})
            )
            robot_count = len(
                usable_set.intersection({9, 10, 11, 12})
            )
            gripper_count = len(
                usable_set.intersection({13, 14, 15})
            )

            row: dict[str, Any] = {
                "timestamp_ms": ts,
                "frame_index": frame_index,
                "detected_marker_count": len(usable_ids),
                "detected_ids": ",".join(map(str, usable_ids)),
                "direct_marker_count": len(direct_ids),
                "direct_ids": ",".join(map(str, direct_ids)),
                "flow_bridged_marker_count": len(flow_ids),
                "flow_bridged_ids": ",".join(map(str, flow_ids)),
                **scene_motion,
                **common_metadata,
                "table_reference_count": table_count,
                "table_reference_at_least_2": table_count >= 2,
                "table_reference_at_least_3": table_count >= 3,
                "table_reference_at_least_4": table_count >= 4,
                "shared_boundary_complete": {3, 4}.issubset(
                    usable_set
                ),
                "robot_arm_any": robot_count >= 1,
                "robot_arm_count": robot_count,
                "gripper_count": gripper_count,
                "gripper_any": gripper_count >= 1,
                "gripper_multiple": gripper_count >= 2,
            }

            for marker_id in range(1, 16):
                row[f"marker_{marker_id}_detected"] = (
                    marker_id in usable_ids
                )
                row[f"marker_{marker_id}_direct"] = (
                    marker_id in direct_ids
                )
                row[f"marker_{marker_id}_flow"] = (
                    marker_id in flow_ids
                )

            frame_rows.append(row)
            sampled += 1
            frame_index += 1

    cap.release()

    detections_df = pd.DataFrame(detection_rows)
    frame_summary_df = pd.DataFrame(frame_rows)
    rates_df = _marker_detection_rate_tables(
        detections_df,
        frame_summary_df,
        config,
    )

    detections_csv = (
        out_dir
        / f"{cfg.output_prefix}_windowed_marker_detections.csv"
    )
    frame_summary_csv = (
        out_dir
        / f"{cfg.output_prefix}_windowed_marker_frame_summary.csv"
    )
    rates_csv = (
        out_dir
        / f"{cfg.output_prefix}_windowed_marker_detection_rates.csv"
    )

    write_csv(detections_df, detections_csv)
    write_csv(frame_summary_df, frame_summary_csv)
    write_csv(rates_df, rates_csv)

    print(f"  sampled trial frames: {len(frame_summary_df)}")
    print(f"  saved: {detections_csv}")
    print(f"  saved: {frame_summary_csv}")
    print(f"  saved: {rates_csv}")

    return (
        detections_df,
        frame_summary_df,
        rates_df,
        detections_csv,
        frame_summary_csv,
        rates_csv,
        props,
    )


def build_dynamic_aoi_per_trial(
    detections_df: pd.DataFrame,
    frame_summary_df: pd.DataFrame,
    config: dict[str, Any],
    out_dir: Path,
    cfg: PipelineConfig,
    processing_windows: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Path,
    Path,
    Path,
    Path,
]:
    """
    Build AOIs independently for every trial.

    All hold, smoothing, and optical-flow-derived states therefore reset at the
    trial boundary. T0, T1, and T2 remain continuous inside a trial.
    """
    trial_root = out_dir / "trial_processing"
    trial_root.mkdir(parents=True, exist_ok=True)

    all_aoi: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    all_trial_availability: list[pd.DataFrame] = []

    for _, window in processing_windows.sort_values(
        "trial_index"
    ).iterrows():
        window_id = int(window["processing_window_id"])
        trial_id = str(window["trial_id"])
        trial_index = int(window["trial_index"])
        task_id = int(window["task_id"])

        trial_detections = detections_df[
            detections_df["processing_window_id"].astype(int).eq(
                window_id
            )
        ].copy()
        trial_frames = frame_summary_df[
            frame_summary_df["processing_window_id"].astype(int).eq(
                window_id
            )
        ].copy()

        trial_dir = trial_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        (
            trial_aoi,
            trial_summary,
            trial_availability,
            _trial_aoi_csv,
            _trial_summary_csv,
            _trial_availability_csv,
        ) = build_dynamic_aoi(
            trial_detections,
            trial_frames,
            config,
            trial_dir,
            cfg,
            f"{cfg.output_prefix}_{trial_id}",
        )

        metadata = {
            "processing_window_id": window_id,
            "trial_id": trial_id,
            "trial_index": trial_index,
            "task_id": task_id,
        }
        for column, value in metadata.items():
            trial_aoi[column] = value
            trial_summary[column] = value
            trial_availability[column] = value

        all_aoi.append(trial_aoi)
        all_summary.append(trial_summary)
        all_trial_availability.append(trial_availability)

    aoi_df = pd.concat(
        all_aoi,
        ignore_index=True,
    ).sort_values(
        ["timestamp_ms", "aoi_name"]
    ).reset_index(drop=True)

    aoi_summary_df = pd.concat(
        all_summary,
        ignore_index=True,
    ).sort_values(
        ["timestamp_ms"]
    ).reset_index(drop=True)

    trial_availability_df = pd.concat(
        all_trial_availability,
        ignore_index=True,
    ).sort_values(
        ["trial_index", "aoi_name"]
    ).reset_index(drop=True)

    overall_availability_df = availability_summary(
        aoi_summary_df
    )

    dynamic_aoi_csv = (
        out_dir / f"{cfg.output_prefix}_windowed_dynamic_aoi.csv"
    )
    dynamic_summary_csv = (
        out_dir
        / f"{cfg.output_prefix}_windowed_dynamic_aoi_summary.csv"
    )
    overall_availability_csv = (
        out_dir
        / f"{cfg.output_prefix}_windowed_aoi_availability.csv"
    )
    trial_availability_csv = (
        out_dir
        / f"{cfg.output_prefix}_trial_aoi_availability.csv"
    )

    write_csv(aoi_df, dynamic_aoi_csv)
    write_csv(aoi_summary_df, dynamic_summary_csv)
    write_csv(
        overall_availability_df,
        overall_availability_csv,
    )
    write_csv(
        trial_availability_df,
        trial_availability_csv,
    )

    print("\n  Combined trial AOI outputs:")
    print(f"    {dynamic_aoi_csv}")
    print(f"    {dynamic_summary_csv}")
    print(f"    {overall_availability_csv}")
    print(f"    {trial_availability_csv}")

    return (
        aoi_df,
        aoi_summary_df,
        overall_availability_df,
        dynamic_aoi_csv,
        dynamic_summary_csv,
        overall_availability_csv,
        trial_availability_csv,
    )


def detect_markers_from_video(
    video_path: Path,
    config: dict[str, Any],
    out_dir: Path,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, dict[str, float]]:
    props = get_video_properties(video_path)
    fps = props["fps"]
    step_frames = max(1, int(round(fps * cfg.sample_every_sec)))
    aruco_dict = get_aruco_dictionary(config["dictionary"])
    cap = cv2.VideoCapture(str(video_path))

    detection_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    frame_index = 0
    sampled = 0

    previous_sample_gray: np.ndarray | None = None
    tracked_states: dict[int, dict[str, Any]] = {}

    print("\n[1/5] Marker detection + short optical-flow bridging")
    print(f"  video: {video_path}")
    print(
        f"  FPS={fps:.3f}, frames={int(props['frame_count'])}, "
        f"size={int(props['width'])}x{int(props['height'])}"
    )
    print(
        f"  sample_every_sec={cfg.sample_every_sec} "
        f"-> every {step_frames} frame(s)"
    )
    print(
        f"  optical flow: enabled={cfg.optical_flow_enabled}, "
        f"robot/gripper max_age={cfg.optical_flow_max_age_ms:.0f} ms, "
        f"table max_age={cfg.table_optical_flow_max_age_ms:.0f} ms; "
        f"head-turn reset={cfg.scene_motion_reset_enabled}"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step_frames == 0:
            ts = int(round(frame_index / fps * 1000.0))
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            scene_motion = estimate_global_scene_motion(
                previous_sample_gray,
                current_gray,
                cfg,
            )
            if bool(scene_motion.get("scene_motion_reset", False)):
                # Do not propagate marker geometry through a head turn, rapid
                # camera sweep, or blurred transition.
                tracked_states = {}

            direct_detections = detect_markers(
                frame,
                aruco_dict,
                cfg.detector_min_perimeter_rate,
            )
            direct_by_id = {
                int(detection["marker_id"]): detection
                for detection in direct_detections
            }

            combined_detections: list[dict[str, Any]] = []
            current_states: dict[int, dict[str, Any]] = {}

            # Direct ArUco detections are always the anchor of truth.
            for marker_id, detection in direct_by_id.items():
                corners = corners_from_detection(detection)
                detection = dict(detection)
                detection.update(
                    {
                        "detection_source": "aruco_direct",
                        "tracking_age_ms": 0.0,
                        "flow_valid_corners": 4,
                        "flow_fb_error_px": 0.0,
                        "flow_area_ratio": 1.0,
                        "flow_center_step_px": 0.0,
                    }
                )
                combined_detections.append(detection)
                current_states[marker_id] = {
                    "corners": corners,
                    "last_direct_ts": ts,
                }

            # Bridge only short gaps. No stale geometry is held after expiry.
            if (
                cfg.optical_flow_enabled
                and previous_sample_gray is not None
                and tracked_states
                and not bool(scene_motion.get("scene_motion_reset", False))
            ):
                for marker_id, state in tracked_states.items():
                    if marker_id in direct_by_id:
                        continue

                    tracking_age_ms = float(ts - int(state["last_direct_ts"]))
                    max_tracking_age_ms = (
                        float(cfg.table_optical_flow_max_age_ms)
                        if int(marker_id) in {1, 2, 3, 4, 5, 6}
                        else float(cfg.optical_flow_max_age_ms)
                    )
                    if tracking_age_ms > max_tracking_age_ms:
                        continue

                    tracked_corners, flow_diag = track_marker_quad_lk(
                        previous_sample_gray,
                        current_gray,
                        np.asarray(state["corners"], dtype=np.float32),
                        cfg,
                    )
                    if tracked_corners is None:
                        continue

                    detection = detection_from_corners(marker_id, tracked_corners)
                    detection.update(
                        {
                            "detection_source": "optical_flow",
                            "tracking_age_ms": tracking_age_ms,
                            **flow_diag,
                        }
                    )
                    combined_detections.append(detection)
                    current_states[marker_id] = {
                        "corners": tracked_corners,
                        "last_direct_ts": int(state["last_direct_ts"]),
                    }

            tracked_states = current_states
            previous_sample_gray = current_gray

            usable_ids = sorted(
                {int(detection["marker_id"]) for detection in combined_detections}
            )
            direct_ids = sorted(direct_by_id)
            flow_ids = sorted(set(usable_ids) - set(direct_ids))

            for detection in combined_detections:
                detection.update(
                    {
                        "source": video_path.name,
                        "frame_index": frame_index,
                        "timestamp_ms": ts,
                        "role": role_for_marker(
                            int(detection["marker_id"]),
                            config,
                        ),
                    }
                )
                detection_rows.append(detection)

            row: dict[str, Any] = {
                "timestamp_ms": ts,
                "frame_index": frame_index,
                "detected_marker_count": len(usable_ids),
                "detected_ids": ",".join(map(str, usable_ids)),
                "direct_marker_count": len(direct_ids),
                "direct_ids": ",".join(map(str, direct_ids)),
                "flow_bridged_marker_count": len(flow_ids),
                "flow_bridged_ids": ",".join(map(str, flow_ids)),
                **scene_motion,
            }
            for marker_id in range(1, 15):
                row[f"marker_{marker_id}_detected"] = marker_id in usable_ids
                row[f"marker_{marker_id}_direct"] = marker_id in direct_ids
                row[f"marker_{marker_id}_flow"] = marker_id in flow_ids

            usable_set = set(usable_ids)
            table_count = len(usable_set.intersection({1, 2, 3, 4, 5, 6}))
            robot_count = len(usable_set.intersection({9, 10, 11, 12}))
            gripper_count = len(usable_set.intersection({13, 14}))
            row.update(
                {
                    "table_reference_count": table_count,
                    "table_reference_at_least_2": table_count >= 2,
                    "table_reference_at_least_3": table_count >= 3,
                    "table_reference_at_least_4": table_count >= 4,
                    "shared_boundary_complete": {3, 4}.issubset(usable_set),
                    "robot_arm_any": robot_count >= 1,
                    "robot_arm_count": robot_count,
                    "gripper_count": gripper_count,
                    "gripper_any": gripper_count >= 1,
                    "gripper_both": gripper_count == 2,
                }
            )
            frame_rows.append(row)
            sampled += 1
            if sampled % 200 == 0:
                print(f"  sampled frames processed: {sampled}")

        frame_index += 1

    cap.release()

    detections_df = pd.DataFrame(detection_rows)
    frame_summary_df = pd.DataFrame(frame_rows)
    detections_csv = out_dir / f"{video_path.stem}_marker_detections_shared6_v6_triangle_robotellipse.csv"
    frame_summary_csv = out_dir / f"{video_path.stem}_marker_frame_summary_shared6_v6_triangle_robotellipse.csv"
    write_csv(detections_df, detections_csv)
    write_csv(frame_summary_df, frame_summary_csv)

    total = len(frame_summary_df)
    rate_rows: list[dict[str, Any]] = []
    for marker_id in range(1, 15):
        usable_col = f"marker_{marker_id}_detected"
        direct_col = f"marker_{marker_id}_direct"
        flow_col = f"marker_{marker_id}_flow"
        usable_count = int(frame_summary_df[usable_col].sum()) if total else 0
        direct_count = int(frame_summary_df[direct_col].sum()) if total else 0
        flow_count = int(frame_summary_df[flow_col].sum()) if total else 0
        rate_rows.append(
            {
                "metric_type": "marker",
                "metric": f"marker_{marker_id}",
                "detected_frames": usable_count,
                "direct_frames": direct_count,
                "flow_bridged_frames": flow_count,
                "sampled_frames": total,
                "rate": usable_count / total if total else np.nan,
                "direct_rate": direct_count / total if total else np.nan,
                "flow_bridge_rate": flow_count / total if total else np.nan,
                "role": role_for_marker(marker_id, config),
            }
        )

    group_cols = {
        "table_reference_at_least_2": "Table mapping usable with similarity fallback",
        "table_reference_at_least_3": "Table mapping usable with affine fallback",
        "table_reference_at_least_4": "Table mapping usable with homography",
        "shared_boundary_complete": "Both shared-boundary markers usable",
        "robot_arm_any": "At least one RobotArm marker usable",
        "gripper_any": "At least one Gripper marker usable",
        "gripper_both": "Both Gripper markers usable",
    }
    for column, description in group_cols.items():
        count = int(frame_summary_df[column].sum()) if total else 0
        rate_rows.append(
            {
                "metric_type": "group",
                "metric": column,
                "detected_frames": count,
                "direct_frames": np.nan,
                "flow_bridged_frames": np.nan,
                "sampled_frames": total,
                "rate": count / total if total else np.nan,
                "direct_rate": np.nan,
                "flow_bridge_rate": np.nan,
                "role": description,
            }
        )

    rates_df = pd.DataFrame(rate_rows)
    rates_csv = out_dir / f"{video_path.stem}_marker_detection_rates_shared6_v6_triangle_robotellipse.csv"
    write_csv(rates_df, rates_csv)

    print(f"  sampled frames: {total}")
    print(f"  saved: {detections_csv}")
    print(f"  saved: {frame_summary_csv}")
    print(f"  saved: {rates_csv}")
    if total:
        display = rates_df[rates_df["metric_type"].eq("group")][
            ["metric", "detected_frames", "sampled_frames", "rate"]
        ]
        print("\n  usable marker-group rates (direct + short optical flow):")
        print(display.to_string(index=False))

    return (
        detections_df,
        frame_summary_df,
        rates_df,
        detections_csv,
        frame_summary_csv,
        rates_csv,
        props,
    )


def to_json_or_empty(obj: Any) -> str:
    return "" if obj is None else json.dumps(obj, ensure_ascii=False)



def add_validated_short_table_holds(
    detections_df: pd.DataFrame,
    frame_summary_df: pd.DataFrame,
    table_marker_ids: list[int],
    cfg: PipelineConfig,
    out_dir: Path,
    stem: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    Fill very short table-marker occlusions using the previous marker geometry,
    but only after validating the gap with the marker position after it
    reappears.

    Acceptance:
      - missing run <= table_validated_hold_max_frames sampled frames
      - no scene-motion reset inside the gap
      - Euclidean displacement between pre-gap and post-gap centres
        <= table_validated_hold_max_shift_px

    Rejection:
      - long gap, head turn, missing post-gap observation, or large endpoint
        displacement. Rejected gaps remain missing, so the AOI falls back to
        the direct three-marker triangle.

    Because processing is offline, this uses both sides of the gap and is safer
    than an unconditional online hold-last rule.
    """
    output_path = (
        out_dir / f"{stem}_table_validated_hold_summary_shared6_v6TH.csv"
    )

    if (
        not cfg.table_validated_hold_enabled
        or detections_df.empty
        or frame_summary_df.empty
        or int(cfg.table_validated_hold_max_frames) <= 0
    ):
        summary = pd.DataFrame(
            [
                {
                    "status": "disabled_or_empty",
                    "accepted_gap_count": 0,
                    "rejected_gap_count": 0,
                    "inserted_marker_rows": 0,
                }
            ]
        )
        write_csv(summary, output_path)
        return detections_df.copy(), summary, output_path

    df = detections_df.copy()
    df["marker_id"] = pd.to_numeric(
        df["marker_id"], errors="coerce"
    ).astype("Int64")
    df["timestamp_ms"] = pd.to_numeric(
        df["timestamp_ms"], errors="coerce"
    ).astype("Int64")
    df = df.dropna(subset=["marker_id", "timestamp_ms"]).copy()
    df["marker_id"] = df["marker_id"].astype(int)
    df["timestamp_ms"] = df["timestamp_ms"].astype(int)

    frame_summary = frame_summary_df.copy()
    frame_summary["timestamp_ms"] = pd.to_numeric(
        frame_summary["timestamp_ms"], errors="coerce"
    ).astype("Int64")
    frame_summary = frame_summary.dropna(subset=["timestamp_ms"]).copy()
    frame_summary["timestamp_ms"] = frame_summary["timestamp_ms"].astype(int)
    frame_summary = frame_summary.sort_values("timestamp_ms").reset_index(
        drop=True
    )

    timestamps = frame_summary["timestamp_ms"].tolist()
    timestamp_to_index = {
        int(timestamp): index
        for index, timestamp in enumerate(timestamps)
    }
    timestamp_to_frame = {
        int(row["timestamp_ms"]): int(row.get("frame_index", -1))
        for _, row in frame_summary.iterrows()
    }
    scene_reset_by_timestamp = {
        int(row["timestamp_ms"]): bool(
            row.get("scene_motion_reset", False)
        )
        for _, row in frame_summary.iterrows()
    }

    existing_pairs = {
        (int(row["marker_id"]), int(row["timestamp_ms"]))
        for _, row in df.iterrows()
    }

    inserted_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []

    for marker_id in table_marker_ids:
        marker_rows = df[
            df["marker_id"].eq(int(marker_id))
        ].copy()
        if marker_rows.empty:
            continue

        # If duplicate rows exist at a timestamp, prefer direct ArUco, then
        # optical flow, then any other source.
        source_priority = {
            "aruco_direct": 0,
            "optical_flow": 1,
            "validated_short_hold": 2,
        }
        marker_rows["_source_priority"] = (
            marker_rows.get(
                "detection_source",
                pd.Series("", index=marker_rows.index),
            )
            .astype(str)
            .map(source_priority)
            .fillna(9)
        )
        marker_rows = (
            marker_rows.sort_values(
                ["timestamp_ms", "_source_priority"]
            )
            .drop_duplicates("timestamp_ms", keep="first")
            .sort_values("timestamp_ms")
            .reset_index(drop=True)
        )

        for row_index in range(len(marker_rows) - 1):
            previous = marker_rows.iloc[row_index]
            following = marker_rows.iloc[row_index + 1]

            previous_ts = int(previous["timestamp_ms"])
            following_ts = int(following["timestamp_ms"])

            if (
                previous_ts not in timestamp_to_index
                or following_ts not in timestamp_to_index
            ):
                continue

            previous_frame_position = timestamp_to_index[previous_ts]
            following_frame_position = timestamp_to_index[following_ts]
            missing_frame_count = (
                following_frame_position
                - previous_frame_position
                - 1
            )

            if missing_frame_count <= 0:
                continue

            gap_timestamps = timestamps[
                previous_frame_position + 1 : following_frame_position
            ]

            previous_center = np.array(
                [
                    float(previous["center_x"]),
                    float(previous["center_y"]),
                ],
                dtype=float,
            )
            following_center = np.array(
                [
                    float(following["center_x"]),
                    float(following["center_y"]),
                ],
                dtype=float,
            )
            endpoint_shift_px = float(
                np.linalg.norm(
                    following_center - previous_center
                )
            )

            has_scene_reset = any(
                scene_reset_by_timestamp.get(int(timestamp), False)
                for timestamp in gap_timestamps
            )

            if missing_frame_count > int(
                cfg.table_validated_hold_max_frames
            ):
                decision = "rejected_gap_too_long"
            elif has_scene_reset:
                decision = "rejected_scene_motion_reset"
            elif endpoint_shift_px > float(
                cfg.table_validated_hold_max_shift_px
            ):
                decision = "rejected_endpoint_shift"
            else:
                decision = "accepted_validated_hold"

            inserted_for_gap = 0

            if decision == "accepted_validated_hold":
                previous_dict = previous.drop(
                    labels=["_source_priority"],
                    errors="ignore",
                ).to_dict()

                for hold_index, gap_timestamp in enumerate(
                    gap_timestamps,
                    start=1,
                ):
                    key = (int(marker_id), int(gap_timestamp))
                    if key in existing_pairs:
                        continue

                    held = dict(previous_dict)
                    held["timestamp_ms"] = int(gap_timestamp)
                    held["frame_index"] = int(
                        timestamp_to_frame.get(
                            int(gap_timestamp),
                            -1,
                        )
                    )
                    held["detection_source"] = (
                        "validated_short_hold"
                    )
                    held["tracking_age_ms"] = float(
                        int(gap_timestamp) - previous_ts
                    )
                    held["gap_hold_frame_index"] = int(hold_index)
                    held["gap_hold_total_frames"] = int(
                        missing_frame_count
                    )
                    held["gap_hold_previous_timestamp_ms"] = int(
                        previous_ts
                    )
                    held["gap_hold_following_timestamp_ms"] = int(
                        following_ts
                    )
                    held["gap_hold_endpoint_shift_px"] = float(
                        endpoint_shift_px
                    )
                    held["gap_hold_max_shift_px"] = float(
                        cfg.table_validated_hold_max_shift_px
                    )
                    inserted_rows.append(held)
                    existing_pairs.add(key)
                    inserted_for_gap += 1

            gap_rows.append(
                {
                    "marker_id": int(marker_id),
                    "previous_timestamp_ms": previous_ts,
                    "following_timestamp_ms": following_ts,
                    "missing_frame_count": int(missing_frame_count),
                    "endpoint_shift_px": endpoint_shift_px,
                    "max_allowed_shift_px": float(
                        cfg.table_validated_hold_max_shift_px
                    ),
                    "scene_motion_reset_in_gap": bool(
                        has_scene_reset
                    ),
                    "decision": decision,
                    "inserted_marker_rows": int(
                        inserted_for_gap
                    ),
                }
            )

    if inserted_rows:
        inserted_df = pd.DataFrame(inserted_rows)
        for column in df.columns:
            if column not in inserted_df.columns:
                inserted_df[column] = np.nan
        for column in inserted_df.columns:
            if column not in df.columns:
                df[column] = np.nan
        df = pd.concat(
            [df, inserted_df[df.columns]],
            ignore_index=True,
        )
        df = df.sort_values(
            ["timestamp_ms", "marker_id"]
        ).reset_index(drop=True)

    gap_summary = pd.DataFrame(gap_rows)
    if gap_summary.empty:
        gap_summary = pd.DataFrame(
            [
                {
                    "marker_id": np.nan,
                    "previous_timestamp_ms": np.nan,
                    "following_timestamp_ms": np.nan,
                    "missing_frame_count": 0,
                    "endpoint_shift_px": np.nan,
                    "max_allowed_shift_px": float(
                        cfg.table_validated_hold_max_shift_px
                    ),
                    "scene_motion_reset_in_gap": False,
                    "decision": "no_short_gaps_found",
                    "inserted_marker_rows": 0,
                }
            ]
        )

    write_csv(gap_summary, output_path)

    accepted_gap_count = int(
        gap_summary["decision"]
        .astype(str)
        .eq("accepted_validated_hold")
        .sum()
    )
    rejected_gap_count = int(
        gap_summary["decision"]
        .astype(str)
        .str.startswith("rejected_")
        .sum()
    )
    inserted_marker_rows = int(
        pd.to_numeric(
            gap_summary["inserted_marker_rows"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    print("\n  Validated short table-marker hold:")
    print(
        f"    max frames={cfg.table_validated_hold_max_frames}, "
        f"max endpoint shift={cfg.table_validated_hold_max_shift_px:.1f} px"
    )
    print(
        f"    accepted gaps={accepted_gap_count}, "
        f"rejected gaps={rejected_gap_count}, "
        f"inserted marker rows={inserted_marker_rows}"
    )
    print(f"    saved: {output_path}")

    return df, gap_summary, output_path


def marker_centers(group_df: pd.DataFrame, marker_ids: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if group_df.empty:
        return out
    for marker_id in marker_ids:
        rows = group_df[group_df["marker_id"].astype(int).eq(int(marker_id))]
        if not rows.empty:
            row = rows.iloc[0]
            out[int(marker_id)] = np.array([float(row["center_x"]), float(row["center_y"])], dtype=float)
    return out



def marker_corner_quads(
    group_df: pd.DataFrame,
    marker_ids: list[int],
) -> dict[int, np.ndarray]:
    """Return marker corner quads in OpenCV ArUco corner order."""
    output: dict[int, np.ndarray] = {}
    if group_df.empty:
        return output

    required = [
        f"corner_{corner_index}_{axis}"
        for corner_index in range(1, 5)
        for axis in ("x", "y")
    ]
    if not all(column in group_df.columns for column in required):
        return output

    for marker_id in marker_ids:
        rows = group_df[
            group_df["marker_id"].astype(int).eq(int(marker_id))
        ]
        if rows.empty:
            continue
        row = rows.iloc[0]
        points = np.array(
            [
                [
                    float(row[f"corner_{corner_index}_x"]),
                    float(row[f"corner_{corner_index}_y"]),
                ]
                for corner_index in range(1, 5)
            ],
            dtype=float,
        )
        if np.isfinite(points).all():
            output[int(marker_id)] = points
    return output


def inverse_transform_points(
    points: np.ndarray,
    mode: str,
    matrix: np.ndarray | None,
) -> np.ndarray:
    """Map image points back into normalized tabletop coordinates."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if matrix is None:
        return np.empty((0, 2), dtype=float)

    if "homography" in str(mode):
        try:
            inverse = np.linalg.inv(np.asarray(matrix, dtype=float))
        except np.linalg.LinAlgError:
            return np.empty((0, 2), dtype=float)
        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),
            inverse.astype(np.float32),
        ).reshape(-1, 2)

    affine = np.asarray(matrix, dtype=float)
    if affine.shape != (2, 3):
        return np.empty((0, 2), dtype=float)
    affine_3x3 = np.vstack([affine, [0.0, 0.0, 1.0]])
    try:
        inverse = np.linalg.inv(affine_3x3)[:2, :]
    except np.linalg.LinAlgError:
        return np.empty((0, 2), dtype=float)
    return cv2.transform(
        points.reshape(-1, 1, 2),
        inverse.astype(np.float32),
    ).reshape(-1, 2)


def calibrate_table_corner_templates(
    direct_quads: dict[int, np.ndarray],
    reference_points: dict[int, np.ndarray],
    mapping_mode: str,
    mapping_matrix: np.ndarray | None,
    existing_templates: dict[int, np.ndarray],
    template_counts: dict[int, int],
    cfg: PipelineConfig,
) -> tuple[dict[int, np.ndarray], dict[int, int], list[int]]:
    """
    Learn each tabletop marker's four corner coordinates in the normalized
    table plane from a high-confidence multi-marker frame.

    The templates remain valid across head turns because they are expressed in
    table coordinates, not image coordinates.
    """
    updated = {
        marker_id: np.asarray(points, dtype=float).copy()
        for marker_id, points in existing_templates.items()
    }
    counts = dict(template_counts)
    accepted_ids: list[int] = []

    if mapping_matrix is None:
        return updated, counts, accepted_ids

    alpha = float(np.clip(cfg.table_corner_template_update_alpha, 0.0, 1.0))

    for marker_id, image_quad in direct_quads.items():
        if marker_id not in reference_points:
            continue

        normalized_quad = inverse_transform_points(
            image_quad,
            mapping_mode,
            mapping_matrix,
        )
        if normalized_quad.shape != (4, 2) or not np.isfinite(normalized_quad).all():
            continue

        normalized_center = normalized_quad.mean(axis=0)
        center_error = float(
            np.linalg.norm(normalized_center - reference_points[marker_id])
        )
        side_lengths = np.linalg.norm(
            np.roll(normalized_quad, -1, axis=0) - normalized_quad,
            axis=1,
        )
        area = polygon_area(normalized_quad.astype(np.float32))

        # Broad plausibility limits; the template is refined across frames.
        if center_error > 0.12:
            continue
        if area <= 1e-5:
            continue
        if np.min(side_lengths) <= 1e-4 or np.max(side_lengths) > 0.35:
            continue
        if not cv2.isContourConvex(normalized_quad.astype(np.float32)):
            continue

        if marker_id in updated and counts.get(marker_id, 0) > 0:
            updated[marker_id] = (
                alpha * normalized_quad
                + (1.0 - alpha) * updated[marker_id]
            )
        else:
            updated[marker_id] = normalized_quad

        counts[marker_id] = counts.get(marker_id, 0) + 1
        accepted_ids.append(marker_id)

    return updated, counts, accepted_ids


def estimate_table_mapping_from_marker_corners(
    current_quads: dict[int, np.ndarray],
    reference_corner_templates: dict[int, np.ndarray],
    reference_points: dict[int, np.ndarray],
    cfg: PipelineConfig,
) -> tuple[
    str,
    np.ndarray | None,
    list[int],
    float,
    float,
]:
    """
    Estimate a table homography from the four corners of at least two directly
    detected, spatially separated markers.

    Returns:
      mode, homography, used_marker_ids, reprojection_error_px,
      maximum_marker_pair_distance_norm
    """
    common_ids = sorted(
        set(current_quads)
        .intersection(reference_corner_templates)
        .intersection(reference_points)
    )
    if len(common_ids) < 2:
        return "corner_unavailable", None, [], np.nan, np.nan

    best_pair_distance = 0.0
    for first, second in itertools.combinations(common_ids, 2):
        distance = float(
            np.linalg.norm(
                reference_points[first] - reference_points[second]
            )
        )
        best_pair_distance = max(best_pair_distance, distance)

    if best_pair_distance < float(cfg.table_corner_min_pair_distance_norm):
        return (
            "corner_pair_too_close",
            None,
            common_ids,
            np.nan,
            best_pair_distance,
        )

    source_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    point_marker_ids: list[int] = []

    for marker_id in common_ids:
        source_quad = np.asarray(
            reference_corner_templates[marker_id],
            dtype=np.float32,
        ).reshape(4, 2)
        target_quad = np.asarray(
            current_quads[marker_id],
            dtype=np.float32,
        ).reshape(4, 2)
        source_blocks.append(source_quad)
        target_blocks.append(target_quad)
        point_marker_ids.extend([marker_id] * 4)

    source = np.vstack(source_blocks)
    target = np.vstack(target_blocks)

    homography, mask = cv2.findHomography(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(
            cfg.table_corner_ransac_reproj_threshold_px
        ),
        maxIters=4000,
        confidence=0.997,
    )
    if homography is None or mask is None or not np.isfinite(homography).all():
        return (
            "corner_homography_failed",
            None,
            common_ids,
            np.nan,
            best_pair_distance,
        )

    inlier_mask = mask.reshape(-1).astype(bool)
    projected = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    point_error = np.linalg.norm(projected - target, axis=1)

    marker_inlier_counts: dict[int, int] = {}
    for marker_id, is_inlier in zip(point_marker_ids, inlier_mask):
        if is_inlier:
            marker_inlier_counts[marker_id] = (
                marker_inlier_counts.get(marker_id, 0) + 1
            )

    used_ids = sorted(
        marker_id
        for marker_id, inlier_count in marker_inlier_counts.items()
        if inlier_count >= 3
    )
    if len(used_ids) < 2:
        return (
            "corner_insufficient_marker_inliers",
            None,
            used_ids,
            np.nan,
            best_pair_distance,
        )

    used_point_mask = np.array(
        [
            is_inlier and marker_id in used_ids
            for marker_id, is_inlier in zip(point_marker_ids, inlier_mask)
        ],
        dtype=bool,
    )
    if not used_point_mask.any():
        return (
            "corner_no_point_inliers",
            None,
            used_ids,
            np.nan,
            best_pair_distance,
        )

    reprojection_error = float(
        np.sqrt(np.mean(point_error[used_point_mask] ** 2))
    )
    if (
        not np.isfinite(reprojection_error)
        or reprojection_error
        > float(cfg.table_corner_max_reprojection_error_px)
    ):
        return (
            "corner_reprojection_error",
            None,
            used_ids,
            reprojection_error,
            best_pair_distance,
        )

    return (
        "corner_homography",
        homography,
        used_ids,
        reprojection_error,
        best_pair_distance,
    )



TABLE_ROW_PAIRS: dict[float, tuple[int, int]] = {
    0.0: (1, 2),
    0.5: (3, 4),
    1.0: (5, 6),
}


def update_table_pair_orientation(
    direct_centers: dict[int, np.ndarray],
    learned_columns: dict[int, int],
) -> tuple[dict[int, int], list[str]]:
    """Infer image-left/image-right within each tabletop marker row pair."""
    updated = dict(learned_columns)
    updates: list[str] = []

    for _, pair in TABLE_ROW_PAIRS.items():
        first, second = pair
        if first not in direct_centers or second not in direct_centers:
            continue

        if float(direct_centers[first][0]) <= float(direct_centers[second][0]):
            left_id, right_id = first, second
        else:
            left_id, right_id = second, first

        if updated.get(left_id) != 0 or updated.get(right_id) != 1:
            updated[left_id] = 0
            updated[right_id] = 1
            updates.append(f"{left_id}=left,{right_id}=right")

    return updated, updates


def build_auto_oriented_table_correspondences(
    current_centers: dict[int, np.ndarray],
    learned_columns: dict[int, int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Convert visible table marker centres into a canonical 2x3 grid while
    ignoring numeric order within each row pair.
    """
    canonical: dict[int, np.ndarray] = {}
    image_points: dict[int, np.ndarray] = {}

    for row_y, pair in TABLE_ROW_PAIRS.items():
        visible = [marker_id for marker_id in pair if marker_id in current_centers]

        if len(visible) == 2:
            visible = sorted(
                visible,
                key=lambda marker_id: float(current_centers[marker_id][0]),
            )
            assignments = {visible[0]: 0, visible[1]: 1}
        else:
            assignments = {
                marker_id: learned_columns[marker_id]
                for marker_id in visible
                if marker_id in learned_columns
            }

        for marker_id, column_x in assignments.items():
            canonical[marker_id] = np.array(
                [float(column_x), float(row_y)],
                dtype=float,
            )
            image_points[marker_id] = np.asarray(
                current_centers[marker_id],
                dtype=float,
            )

    return canonical, image_points



TABLE_REGION_MARKER_IDS: dict[str, list[int]] = {
    "Delivery": [1, 2, 3, 4],
    "Task": [3, 4, 5, 6],
}


TABLE_RAY_LAYOUT: dict[str, dict[str, int]] = {
    # Actual pilot layout:
    #
    #   5 ----- 3 ----- 1
    #      Task | Delivery
    #   6 ----- 4 ----- 2
    #
    # Task boundaries:     5 -> 6 and 3 -> 4
    # Delivery boundaries: 3 -> 4 and 1 -> 2
    "Task": {
        "top_left": 5,
        "top_right": 3,
        "bottom_left": 6,
        "bottom_right": 4,
    },
    "Delivery": {
        "top_left": 3,
        "top_right": 1,
        "bottom_left": 4,
        "bottom_right": 2,
    },
}


def expand_table_region_marker_centers(
    current_centers: dict[int, np.ndarray],
    found_ids: list[int],
    region_name: str,
    expansion_px: float,
) -> dict[int, np.ndarray]:
    """
    Conservative fallback expansion for the three-marker triangle.

    Delivery markers 1/2 are shifted away from shared markers 3/4.
    Task markers 5/6 are shifted away from shared markers 3/4.
    Shared markers 3/4 remain fixed.
    """
    expanded = {
        int(marker_id): np.asarray(
            current_centers[int(marker_id)],
            dtype=float,
        ).copy()
        for marker_id in found_ids
    }

    amount = max(float(expansion_px), 0.0)
    if amount <= 0.0:
        return expanded

    outer_ids = (
        {1, 2}
        if region_name == "Delivery"
        else {5, 6}
        if region_name == "Task"
        else set()
    )
    if not outer_ids:
        return expanded

    shared_ids = [
        marker_id
        for marker_id in (3, 4)
        if marker_id in expanded
    ]
    if not shared_ids:
        return expanded

    shared_anchor = np.mean(
        [expanded[marker_id] for marker_id in shared_ids],
        axis=0,
    )

    for marker_id in sorted(
        set(found_ids).intersection(outer_ids)
    ):
        point = expanded[marker_id]
        direction = point - shared_anchor
        norm = float(np.linalg.norm(direction))
        if np.isfinite(norm) and norm >= 1e-6:
            expanded[marker_id] = (
                point + amount * direction / norm
            )

    return expanded


def extend_boundary_ray_to_y(
    top_point: np.ndarray,
    lower_point: np.ndarray,
    target_y: float,
) -> np.ndarray | None:
    """
    Extend the directed line top -> lower until y == target_y.

    A valid tabletop boundary should move downward in image coordinates.
    """
    top = np.asarray(top_point, dtype=float)
    lower = np.asarray(lower_point, dtype=float)
    dy = float(lower[1] - top[1])

    if not np.isfinite(dy) or dy <= 5.0:
        return None

    scale = float(target_y - top[1]) / dy
    if not np.isfinite(scale) or scale < 1.0:
        return None

    point = top + scale * (lower - top)
    if not np.isfinite(point).all():
        return None
    return point


def clip_convex_polygon_to_frame(
    polygon: np.ndarray,
    frame_width: float,
    frame_height: float,
) -> np.ndarray | None:
    """
    Intersect a convex polygon with the image rectangle.
    """
    width = float(frame_width)
    height = float(frame_height)
    if width <= 1.0 or height <= 1.0:
        return None

    points = np.asarray(
        polygon,
        dtype=np.float32,
    ).reshape(-1, 2)
    if len(points) < 3:
        return None

    hull = cv2.convexHull(
        points.reshape(-1, 1, 2),
        clockwise=False,
        returnPoints=True,
    ).reshape(-1, 2)

    frame_polygon = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )

    try:
        intersection_area, intersection = (
            cv2.intersectConvexConvex(
                hull.astype(np.float32),
                frame_polygon,
            )
        )
    except cv2.error:
        return None

    if (
        intersection is None
        or float(intersection_area) <= 0.0
    ):
        return None

    clipped = np.asarray(
        intersection,
        dtype=np.float32,
    ).reshape(-1, 2)

    if len(clipped) < 3:
        return None

    clipped = cv2.convexHull(
        clipped.reshape(-1, 1, 2),
        clockwise=False,
        returnPoints=True,
    ).reshape(-1, 2)
    return clipped


def build_ray_region_polygon(
    current_centers: dict[int, np.ndarray],
    region_name: str,
    frame_width: float,
    frame_height: float,
    bottom_margin_px: float,
) -> tuple[np.ndarray | None, list[int], str]:
    """
    Build a Task or Delivery region from two downward boundary rays.

    Task:
      top edge 5--3
      left ray 5->6
      right/shared ray 3->4

    Delivery:
      top edge 3--1
      left/shared ray 3->4
      right ray 1->2

    Both rays are extended to the image bottom. The resulting polygon is
    clipped to the visible frame.
    """
    layout = TABLE_RAY_LAYOUT.get(region_name)
    if layout is None:
        return None, [], "unknown_region"

    expected_ids = list(layout.values())
    found_ids = [
        marker_id
        for marker_id in expected_ids
        if marker_id in current_centers
    ]
    if len(found_ids) != 4:
        return None, found_ids, "ray_requires_4_markers"

    target_y = max(
        0.0,
        float(frame_height)
        - 1.0
        - max(float(bottom_margin_px), 0.0),
    )

    top_left = np.asarray(
        current_centers[layout["top_left"]],
        dtype=float,
    )
    top_right = np.asarray(
        current_centers[layout["top_right"]],
        dtype=float,
    )
    bottom_left = np.asarray(
        current_centers[layout["bottom_left"]],
        dtype=float,
    )
    bottom_right = np.asarray(
        current_centers[layout["bottom_right"]],
        dtype=float,
    )

    extended_left = extend_boundary_ray_to_y(
        top_left,
        bottom_left,
        target_y,
    )
    extended_right = extend_boundary_ray_to_y(
        top_right,
        bottom_right,
        target_y,
    )

    if extended_left is None or extended_right is None:
        return None, found_ids, "ray_direction_invalid"

    raw_polygon = np.array(
        [
            top_left,
            top_right,
            extended_right,
            extended_left,
        ],
        dtype=np.float32,
    )

    polygon = clip_convex_polygon_to_frame(
        raw_polygon,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if polygon is None:
        return None, found_ids, "ray_clip_failed"

    if polygon_area(polygon) < 800.0:
        return None, found_ids, "ray_polygon_too_small"

    return (
        polygon,
        found_ids,
        "ray_quad_4_to_frame_bottom",
    )


def build_direct_region_hull(
    current_centers: dict[int, np.ndarray],
    expected_marker_ids: list[int],
    region_name: str = "",
    outer_expansion_px: float = 0.0,
    frame_width: float | None = None,
    frame_height: float | None = None,
    ray_bottom_margin_px: float = 0.0,
) -> tuple[np.ndarray | None, list[int], str]:
    """
    Region construction policy:

    4 visible markers:
      Use the correct downward-ray region and extend it to the image bottom.

    Exactly 3 visible markers:
      Use a conservative direct triangle. No missing corner is inferred.

    Fewer than 3:
      Unavailable.
    """
    found_ids = [
        marker_id
        for marker_id in expected_marker_ids
        if marker_id in current_centers
    ]

    if len(found_ids) < 3:
        return None, found_ids, "unavailable"

    if (
        len(found_ids) == 4
        and frame_width is not None
        and frame_height is not None
    ):
        ray_polygon, ray_ids, ray_mode = (
            build_ray_region_polygon(
                current_centers=current_centers,
                region_name=region_name,
                frame_width=float(frame_width),
                frame_height=float(frame_height),
                bottom_margin_px=float(
                    ray_bottom_margin_px
                ),
            )
        )
        if ray_polygon is not None:
            return ray_polygon, ray_ids, ray_mode

    # Conservative fallback, especially for exactly three visible markers.
    expanded_centers = expand_table_region_marker_centers(
        current_centers=current_centers,
        found_ids=found_ids,
        region_name=region_name,
        expansion_px=outer_expansion_px,
    )

    points = np.array(
        [
            expanded_centers[marker_id]
            for marker_id in found_ids
        ],
        dtype=np.float32,
    ).reshape(-1, 2)

    hull = cv2.convexHull(
        points.reshape(-1, 1, 2),
        clockwise=False,
        returnPoints=True,
    ).reshape(-1, 2)

    suffix = (
        f"_outer_expanded_{float(outer_expansion_px):g}px"
        if float(outer_expansion_px) > 0.0
        else ""
    )

    if len(found_ids) == 3:
        if hull.shape != (3, 2):
            return None, found_ids, "degenerate_triangle"
        if polygon_area(hull) < 800.0:
            return None, found_ids, "triangle_too_small"
        return (
            hull,
            found_ids,
            "triangle_3_direct" + suffix,
        )

    if hull.shape != (4, 2):
        return None, found_ids, "degenerate_quadrilateral"
    if polygon_area(hull) < 800.0:
        return None, found_ids, "quadrilateral_too_small"

    return (
        hull,
        found_ids,
        "quadrilateral_4_fallback" + suffix,
    )


def estimate_simple_table_mapping(
    canonical_points: dict[int, np.ndarray],
    image_points: dict[int, np.ndarray],
    cfg: PipelineConfig,
) -> tuple[str, np.ndarray | None, list[int], float]:
    """
    Pilot-simple tabletop mapping:
      >=4 points -> homography RANSAC
      >=3 non-collinear points -> affine RANSAC
      2 well-separated points -> similarity fallback
    """
    marker_ids = sorted(set(canonical_points).intersection(image_points))
    if len(marker_ids) < 2:
        return "unavailable", None, marker_ids, np.nan

    source = np.array(
        [canonical_points[marker_id] for marker_id in marker_ids],
        dtype=np.float32,
    )
    target = np.array(
        [image_points[marker_id] for marker_id in marker_ids],
        dtype=np.float32,
    )
    unique_rows = len(set(np.round(source[:, 1], 4).tolist()))
    unique_columns = len(set(np.round(source[:, 0], 4).tolist()))

    if len(marker_ids) >= 4 and unique_rows >= 2 and unique_columns >= 2:
        homography, mask = cv2.findHomography(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=12.0,
            maxIters=3000,
            confidence=0.995,
        )
        if (
            homography is not None
            and mask is not None
            and np.isfinite(homography).all()
        ):
            inliers = mask.reshape(-1).astype(bool)
            used_ids = [
                marker_id
                for marker_id, is_inlier in zip(marker_ids, inliers)
                if is_inlier
            ]
            if len(used_ids) >= 4:
                projected = cv2.perspectiveTransform(
                    source.reshape(-1, 1, 2),
                    homography,
                ).reshape(-1, 2)
                errors = np.linalg.norm(projected - target, axis=1)
                rms = float(np.sqrt(np.mean(errors[inliers] ** 2)))
                return "homography_ransac", homography, used_ids, rms

    if len(marker_ids) >= 3:
        non_collinear = any(
            abs(float(np.cross(source[j] - source[i], source[k] - source[i])))
            > 1e-4
            for i, j, k in itertools.combinations(range(len(source)), 3)
        )
        if non_collinear:
            affine, mask = cv2.estimateAffine2D(
                source,
                target,
                method=cv2.RANSAC,
                ransacReprojThreshold=12.0,
                maxIters=3000,
                confidence=0.995,
                refineIters=20,
            )
            if (
                affine is not None
                and mask is not None
                and np.isfinite(affine).all()
            ):
                inliers = mask.reshape(-1).astype(bool)
                used_ids = [
                    marker_id
                    for marker_id, is_inlier in zip(marker_ids, inliers)
                    if is_inlier
                ]
                if len(used_ids) >= 3:
                    projected = cv2.transform(
                        source.reshape(-1, 1, 2),
                        affine,
                    ).reshape(-1, 2)
                    errors = np.linalg.norm(projected - target, axis=1)
                    rms = float(np.sqrt(np.mean(errors[inliers] ** 2)))
                    return "affine_ransac", affine, used_ids, rms

    best_pair: tuple[int, int] | None = None
    best_distance = -1.0
    for first, second in itertools.combinations(marker_ids, 2):
        canonical_distance = float(
            np.linalg.norm(canonical_points[second] - canonical_points[first])
        )
        image_distance = float(
            np.linalg.norm(image_points[second] - image_points[first])
        )
        if (
            canonical_distance >= 0.75
            and image_distance >= 80.0
            and canonical_distance > best_distance
        ):
            best_pair = (first, second)
            best_distance = canonical_distance

    if best_pair is not None:
        first, second = best_pair
        src_first = canonical_points[first]
        src_second = canonical_points[second]
        dst_first = image_points[first]
        dst_second = image_points[second]

        src_vector = src_second - src_first
        dst_vector = dst_second - dst_first
        src_length = float(np.linalg.norm(src_vector))
        dst_length = float(np.linalg.norm(dst_vector))

        if src_length > 1e-6 and dst_length > 1e-6:
            scale = dst_length / src_length
            theta = (
                float(np.arctan2(dst_vector[1], dst_vector[0]))
                - float(np.arctan2(src_vector[1], src_vector[0]))
            )
            cosine = float(np.cos(theta))
            sine = float(np.sin(theta))
            linear = scale * np.array(
                [[cosine, -sine], [sine, cosine]],
                dtype=float,
            )
            translation = dst_first - linear @ src_first
            affine = np.column_stack([linear, translation]).astype(np.float32)
            return "similarity_2", affine, [first, second], 0.0

    return "unavailable", None, marker_ids, np.nan


def simple_table_polygons_are_valid(
    polygons: dict[str, np.ndarray],
) -> bool:
    """Essential checks only; no temporal rejection logic."""
    for polygon in polygons.values():
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if points.shape != (4, 2):
            return False
        if not np.isfinite(points).all():
            return False
        if polygon_area(points) < 800.0:
            return False
        if not cv2.isContourConvex(points):
            return False
    return True


def connected_ellipse_polygon(
    first_center: np.ndarray,
    second_center: np.ndarray,
    endpoint_radius_px: float,
    point_count: int = 40,
) -> np.ndarray:
    """Create one rotated ellipse spanning both gripper marker centres."""
    first = np.asarray(first_center, dtype=float)
    second = np.asarray(second_center, dtype=float)
    midpoint = (first + second) / 2.0
    delta = second - first
    separation = float(np.linalg.norm(delta))
    angle = float(np.arctan2(delta[1], delta[0])) if separation > 1e-6 else 0.0

    major_radius = separation / 2.0 + float(endpoint_radius_px)
    minor_radius = float(endpoint_radius_px)

    parameter = np.linspace(
        0.0,
        2.0 * np.pi,
        int(point_count),
        endpoint=False,
    )
    local = np.column_stack(
        [
            major_radius * np.cos(parameter),
            minor_radius * np.sin(parameter),
        ]
    )
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )
    return local @ rotation.T + midpoint




def oriented_ellipse_polygon(
    center: np.ndarray,
    major_radius_px: float,
    minor_radius_px: float,
    angle_rad: float,
    point_count: int = 48,
) -> np.ndarray:
    """Create one rotated ellipse centred on a single marker."""
    center = np.asarray(center, dtype=float).reshape(2)
    major = float(major_radius_px)
    minor = float(minor_radius_px)

    parameter = np.linspace(
        0.0,
        2.0 * np.pi,
        int(point_count),
        endpoint=False,
    )
    local = np.column_stack(
        [
            major * np.cos(parameter),
            minor * np.sin(parameter),
        ]
    )

    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )
    return local @ rotation.T + center


def marker_local_x_angle_rad(
    group_df: pd.DataFrame,
    marker_id: int,
) -> float:
    """
    Estimate marker orientation from corner 1 -> corner 2.

    This direction becomes the major axis of the single-marker Gripper ellipse.
    With optical-flow detections, the tracked marker corners are used.
    """
    rows = group_df[
        group_df["marker_id"].astype(int).eq(int(marker_id))
    ]
    if rows.empty:
        return 0.0

    row = rows.iloc[0]
    required = [
        "corner_1_x",
        "corner_1_y",
        "corner_2_x",
        "corner_2_y",
    ]
    if any(column not in row.index for column in required):
        return 0.0

    first = np.array(
        [float(row["corner_1_x"]), float(row["corner_1_y"])],
        dtype=float,
    )
    second = np.array(
        [float(row["corner_2_x"]), float(row["corner_2_y"])],
        dtype=float,
    )
    delta = second - first
    if not np.isfinite(delta).all() or float(np.linalg.norm(delta)) < 1e-6:
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))



def gripper_arm_axis_angle_rad(
    robot_centers: dict[int, np.ndarray],
    gripper_centers: dict[int, np.ndarray],
    group_df: pd.DataFrame,
    fallback_marker_id: int | None,
    marker_angle_offset_deg: float,
) -> tuple[float, str]:
    """
    Determine the Gripper AOI major-axis direction.

    Priority:
      1. RobotArm centroid -> Gripper centroid.
      2. Gripper marker local axis rotated by the configured offset
         (default 90 degrees, corresponding to marker local-y).
      3. Screen vertical.
    """
    if robot_centers and gripper_centers:
        robot_center = np.mean(
            [
                np.asarray(center, dtype=float)
                for center in robot_centers.values()
            ],
            axis=0,
        )
        gripper_center = np.mean(
            [
                np.asarray(center, dtype=float)
                for center in gripper_centers.values()
            ],
            axis=0,
        )
        delta = gripper_center - robot_center
        norm = float(np.linalg.norm(delta))
        if np.isfinite(norm) and norm >= 1e-6:
            return (
                float(np.arctan2(delta[1], delta[0])),
                "robotarm_centroid_to_gripper_centroid",
            )

    if fallback_marker_id is not None:
        marker_angle = marker_local_x_angle_rad(
            group_df,
            int(fallback_marker_id),
        )
        return (
            marker_angle
            + np.deg2rad(float(marker_angle_offset_deg)),
            "gripper_marker_local_y_fallback",
        )

    return float(np.pi / 2.0), "screen_vertical_fallback"


def arm_aligned_gripper_ellipse_polygon(
    centers: list[np.ndarray],
    axis_angle_rad: float,
    envelope_radius_px: float,
    minimum_major_radius_px: float,
    minimum_minor_radius_px: float,
    point_count: int = 64,
) -> np.ndarray:
    """
    Enclose one or more Gripper markers with an ellipse whose long axis follows
    the robot-arm direction.

    Marker spread is measured in the arm-aligned coordinate system. The
    resulting ellipse is kept elongated along that axis.
    """
    points = np.asarray(centers, dtype=float).reshape(-1, 2)
    if len(points) < 1:
        raise ValueError("At least one Gripper centre is required.")

    center = points.mean(axis=0)
    cosine = float(np.cos(axis_angle_rad))
    sine = float(np.sin(axis_angle_rad))
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )

    local = (points - center) @ rotation
    axial_extent = float(np.max(np.abs(local[:, 0])))
    lateral_extent = float(np.max(np.abs(local[:, 1])))

    envelope = max(float(envelope_radius_px), 0.0)
    major_radius = max(
        float(minimum_major_radius_px),
        axial_extent + envelope,
    )
    minor_radius = max(
        float(minimum_minor_radius_px),
        lateral_extent + envelope,
    )

    # Keep the long axis along the arm direction.
    major_radius = max(
        major_radius,
        minor_radius * 1.15,
    )

    return oriented_ellipse_polygon(
        center=center,
        major_radius_px=major_radius,
        minor_radius_px=minor_radius,
        angle_rad=axis_angle_rad,
        point_count=point_count,
    )


def multi_marker_connected_ellipse_polygon(
    centers: list[np.ndarray],
    endpoint_radius_px: float,
    point_count: int = 64,
    boundary_samples_per_marker: int = 24,
) -> np.ndarray:
    """
    Create one rotated ellipse enclosing 2–4 marker-centred circles.

    Two centres:
        Uses the same connected-ellipse construction as the Gripper.

    Three or more centres:
        Uses the principal direction of all centres, builds a local bounding
        ellipse, and enlarges it until sampled marker-circle boundaries are
        enclosed.

    The existing RobotArm marker radius is used as the envelope thickness.
    """
    points = np.asarray(centers, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        raise ValueError("At least two centres are required.")

    radius = float(endpoint_radius_px)

    if len(points) == 2:
        return connected_ellipse_polygon(
            points[0],
            points[1],
            radius,
            point_count=point_count,
        )

    centroid = points.mean(axis=0)
    centered = points - centroid

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]

    # Keep a deterministic right-handed 2D coordinate system.
    if np.linalg.det(axes) < 0:
        axes[:, 1] *= -1.0

    local_centers = centered @ axes
    local_min = local_centers.min(axis=0)
    local_max = local_centers.max(axis=0)
    local_midpoint = (local_min + local_max) / 2.0
    shifted_centers = local_centers - local_midpoint

    base_major = max(
        float(np.max(np.abs(shifted_centers[:, 0]))) + radius,
        radius,
    )
    base_minor = max(
        float(np.max(np.abs(shifted_centers[:, 1]))) + radius,
        radius,
    )

    # Sample the boundary of each marker circle and enlarge the ellipse just
    # enough to contain those samples.
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        int(boundary_samples_per_marker),
        endpoint=False,
    )
    circle_offsets_world = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles)]
    )
    circle_offsets_local = circle_offsets_world @ axes

    envelope_points = np.vstack(
        [
            shifted_center + circle_offsets_local
            for shifted_center in shifted_centers
        ]
    )

    normalized_radius = np.sqrt(
        (envelope_points[:, 0] / base_major) ** 2
        + (envelope_points[:, 1] / base_minor) ** 2
    )
    scale = max(float(np.max(normalized_radius)), 1.0) * 1.02

    major_radius = base_major * scale
    minor_radius = base_minor * scale

    parameter = np.linspace(
        0.0,
        2.0 * np.pi,
        int(point_count),
        endpoint=False,
    )
    ellipse_local = np.column_stack(
        [
            major_radius * np.cos(parameter),
            minor_radius * np.sin(parameter),
        ]
    )

    ellipse_center_world = centroid + local_midpoint @ axes.T
    return ellipse_local @ axes.T + ellipse_center_world


def choose_non_collinear_triplet(ids: list[int], points_norm: dict[int, np.ndarray]) -> list[int] | None:
    best: list[int] | None = None
    best_area = 0.0
    for combo in itertools.combinations(ids, 3):
        a, b, c = [points_norm[i] for i in combo]
        area = abs(float(np.cross(b - a, c - a))) / 2.0
        if area > best_area:
            best_area = area
            best = list(combo)
    return best if best_area > 1e-5 else None


def estimate_table_mapping(
    current_centers: dict[int, np.ndarray],
    ref_points_norm: dict[int, np.ndarray],
    cfg: PipelineConfig,
) -> tuple[str, np.ndarray | None, list[int]]:
    """
    Robustly estimate the table mapping.

    >=4 markers:
        Homography with RANSAC. Only inlier marker IDs are used for
        reprojection-quality checks.

    >=3 markers:
        Affine transform with RANSAC.

    2 markers:
        Similarity transform using the most widely separated pair.

    A single mistracked marker therefore does not invalidate the whole
    Delivery/Task AOI.
    """
    ids = sorted(set(current_centers).intersection(ref_points_norm))

    if len(ids) >= 4:
        src = np.array([ref_points_norm[i] for i in ids], dtype=np.float32)
        dst = np.array([current_centers[i] for i in ids], dtype=np.float32)
        H, mask = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(cfg.table_ransac_reproj_threshold_px),
            maxIters=3000,
            confidence=0.995,
        )
        if H is not None and np.isfinite(H).all() and mask is not None:
            inlier_flags = mask.reshape(-1).astype(bool)
            inlier_ids = [marker_id for marker_id, ok in zip(ids, inlier_flags) if ok]
            if len(inlier_ids) >= 4:
                return "homography_ransac", H, inlier_ids

    if len(ids) >= 3:
        src = np.array([ref_points_norm[i] for i in ids], dtype=np.float32)
        dst = np.array([current_centers[i] for i in ids], dtype=np.float32)
        A, mask = cv2.estimateAffine2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(cfg.table_ransac_reproj_threshold_px),
            maxIters=3000,
            confidence=0.995,
            refineIters=20,
        )
        if A is not None and np.isfinite(A).all() and mask is not None:
            inlier_flags = mask.reshape(-1).astype(bool)
            inlier_ids = [marker_id for marker_id, ok in zip(ids, inlier_flags) if ok]
            if len(inlier_ids) >= 3:
                return "affine_ransac", A, inlier_ids

    if len(ids) >= 2:
        best_pair = None
        best_distance = -1.0
        for i, j in itertools.combinations(ids, 2):
            distance = float(np.linalg.norm(ref_points_norm[j] - ref_points_norm[i]))
            if distance > best_distance:
                best_distance = distance
                best_pair = (i, j)

        if best_pair is not None and best_distance >= 0.75:
            i, j = best_pair
            src_1 = np.asarray(ref_points_norm[i], dtype=float)
            src_2 = np.asarray(ref_points_norm[j], dtype=float)
            dst_1 = np.asarray(current_centers[i], dtype=float)
            dst_2 = np.asarray(current_centers[j], dtype=float)

            src_vec = src_2 - src_1
            dst_vec = dst_2 - dst_1
            src_len = float(np.linalg.norm(src_vec))
            dst_len = float(np.linalg.norm(dst_vec))

            if src_len > 1e-6 and dst_len > 1e-6:
                scale = dst_len / src_len
                theta = (
                    float(np.arctan2(dst_vec[1], dst_vec[0]))
                    - float(np.arctan2(src_vec[1], src_vec[0]))
                )
                cos_t = float(np.cos(theta))
                sin_t = float(np.sin(theta))
                linear = scale * np.array(
                    [[cos_t, -sin_t], [sin_t, cos_t]],
                    dtype=float,
                )
                translation = dst_1 - linear @ src_1
                A = np.column_stack([linear, translation]).astype(np.float32)
                if np.isfinite(A).all():
                    return "similarity_2", A, [i, j]

    return "unavailable", None, ids

def transform_points(points: np.ndarray, mode: str, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if mode.startswith("homography"):
        return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2).astype(float)
    return cv2.transform(pts, matrix).reshape(-1, 2).astype(float)


def normalized_table_polygons(delivery_scale: float, task_scale: float) -> dict[str, np.ndarray]:
    d_left = 0.5 + (0.0 - 0.5) * float(delivery_scale)
    d_right = 0.5 + (1.0 - 0.5) * float(delivery_scale)
    d_top = 0.5 + (0.0 - 0.5) * float(delivery_scale)
    t_left = 0.5 + (0.0 - 0.5) * float(task_scale)
    t_right = 0.5 + (1.0 - 0.5) * float(task_scale)
    t_bottom = 0.5 + (1.0 - 0.5) * float(task_scale)
    # Both polygons share the same y=0.5 boundary. Use the broader x range so
    # the boundary remains identical and no gap is introduced.
    shared_left = min(d_left, t_left)
    shared_right = max(d_right, t_right)
    return {
        "Delivery": np.array([
            [d_left, d_top], [d_right, d_top], [shared_right, 0.5], [shared_left, 0.5]
        ], dtype=float),
        "Task": np.array([
            [shared_left, 0.5], [shared_right, 0.5], [t_right, t_bottom], [t_left, t_bottom]
        ], dtype=float),
    }


def order_points_around_centroid(points: list[list[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def scale_polygon(points: np.ndarray, scale: float) -> np.ndarray:
    center = points.mean(axis=0)
    return center + (points - center) * float(scale)


def translate_geometry(points: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) + np.asarray(displacement, dtype=float)


def median_displacement(
    current: dict[int, np.ndarray],
    previous: dict[int, np.ndarray],
) -> np.ndarray | None:
    common = sorted(set(current).intersection(previous))
    if not common:
        return None
    displacements = np.array([current[i] - previous[i] for i in common], dtype=float)
    return np.median(displacements, axis=0)


def gripper_geometry(
    group_df: pd.DataFrame,
    ids: list[int],
    double_base: float,
    single_radius: float,
    max_radius: float,
) -> tuple[bool, list[float] | None, float | None, list[int], str]:
    centers = marker_centers(group_df, ids)
    found = sorted(centers)
    if not found:
        return False, None, None, [], "unavailable"
    pts = np.array([centers[i] for i in found], dtype=float)
    if len(found) == 1:
        return True, pts[0].tolist(), float(single_radius), found, f"single_{found[0]}"
    center = pts.mean(axis=0)
    radius = min(float(double_base) + float(np.max(np.linalg.norm(pts - center, axis=1))), float(max_radius))
    return True, center.tolist(), radius, found, "double"


def availability_summary(aoi_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if aoi_summary_df.empty:
        return pd.DataFrame()
    timestamps = aoi_summary_df["timestamp_ms"].astype(int).tolist()
    n = len(aoi_summary_df)
    for aoi in AVAILABILITY_AOIS:
        col = f"{aoi}_available"
        if col not in aoi_summary_df.columns:
            continue
        flags = aoi_summary_df[col].astype(bool).tolist()
        rows.append({
            "aoi_name": aoi,
            "available_frames": int(sum(flags)),
            "sampled_frames": n,
            "availability_rate": float(sum(flags) / n) if n else np.nan,
            "longest_unavailable_run_ms": longest_false_run_ms(flags, timestamps),
        })
    return pd.DataFrame(rows)



def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return abs(float(cv2.contourArea(pts)))


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    return pts.mean(axis=0)


def mapping_reprojection_error(
    current_centers: dict[int, np.ndarray],
    reference_points: dict[int, np.ndarray],
    mode: str,
    matrix: np.ndarray | None,
    used_ids: list[int] | None = None,
) -> float:
    if matrix is None:
        return np.nan
    common = sorted(set(current_centers).intersection(reference_points))
    if used_ids is not None:
        common = [marker_id for marker_id in common if marker_id in set(used_ids)]
    if not common:
        return np.nan
    source = np.array([reference_points[i] for i in common], dtype=float)
    target = np.array([current_centers[i] for i in common], dtype=float)
    projected = transform_points(source, mode, matrix)
    if projected.shape != target.shape:
        return np.nan
    return float(np.sqrt(np.mean(np.sum((projected - target) ** 2, axis=1))))


def validate_table_geometry(
    candidate: dict[str, np.ndarray],
    previous: dict[str, np.ndarray] | None,
    previous_ts: int | None,
    current_ts: int,
    reprojection_error_px: float,
    cfg: PipelineConfig,
) -> tuple[bool, str]:
    for name, polygon in candidate.items():
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3 or not np.isfinite(pts).all():
            return False, f"{name}_nonfinite"
        if polygon_area(pts) < float(cfg.table_min_polygon_area_px2):
            return False, f"{name}_area_too_small"
        if not cv2.isContourConvex(pts):
            return False, f"{name}_not_convex"

    if np.isfinite(reprojection_error_px) and (
        reprojection_error_px > float(cfg.table_max_reprojection_error_px)
    ):
        return False, "reprojection_error"

    recent_previous = (
        previous is not None
        and previous_ts is not None
        and (current_ts - previous_ts) <= float(cfg.temporal_quality_window_ms)
    )
    if recent_previous:
        for name in candidate:
            old_area = polygon_area(previous[name])
            new_area = polygon_area(candidate[name])
            area_ratio = new_area / old_area if old_area > 0 else np.nan
            if not np.isfinite(area_ratio) or not (
                float(cfg.table_min_area_ratio)
                <= area_ratio
                <= float(cfg.table_max_area_ratio)
            ):
                return False, f"{name}_area_jump"

            centroid_jump = float(
                np.linalg.norm(
                    polygon_centroid(candidate[name])
                    - polygon_centroid(previous[name])
                )
            )
            if centroid_jump > float(cfg.table_max_centroid_jump_px):
                return False, f"{name}_centroid_jump"

    return True, "accepted"


def smooth_points(
    current: np.ndarray,
    previous: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    if previous is None:
        return current
    previous = np.asarray(previous, dtype=float)
    if current.shape != previous.shape:
        return current
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * current + (1.0 - alpha) * previous


def detection_source_for_marker(group_df: pd.DataFrame, marker_id: int) -> str:
    rows = group_df[group_df["marker_id"].astype(int).eq(int(marker_id))]
    if rows.empty or "detection_source" not in rows.columns:
        return "unknown"
    return str(rows.iloc[0]["detection_source"])


def build_dynamic_aoi(
    detections_df: pd.DataFrame,
    frame_summary_df: pd.DataFrame,
    config: dict[str, Any],
    out_dir: Path,
    cfg: PipelineConfig,
    stem: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    print("\n[2/5] Dynamic AOI — direct triangle geometry with forward-backward validated short hold v6TH")
    df = detections_df.copy()
    if not df.empty:
        df["marker_id"] = pd.to_numeric(df["marker_id"], errors="coerce").astype("Int64")
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["marker_id", "timestamp_ms"]).copy()
        df["marker_id"] = df["marker_id"].astype(int)
        df["timestamp_ms"] = df["timestamp_ms"].astype(int)

    table_ids = [1, 2, 3, 4, 5, 6]

    df, validated_hold_summary_df, validated_hold_summary_csv = (
        add_validated_short_table_holds(
            df,
            frame_summary_df,
            table_ids,
            cfg,
            out_dir,
            stem,
        )
    )

    robot_ids = [9, 10, 11, 12]
    gripper_ids = [13, 14, 15]
    reference_points = {
        int(key): np.asarray(value, dtype=float)
        for key, value in config["table_reference_points_normalized"].items()
    }
    normalized_polygons = normalized_table_polygons(
        cfg.delivery_scale,
        cfg.task_scale,
    )

    aoi_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    last_static_polygons: dict[str, np.ndarray] | None = None
    last_static_ts: int | None = None
    table_pair_columns: dict[int, int] = {}
    last_robot_centers: dict[int, np.ndarray] = {}
    last_robot_ts: dict[int, int] = {}
    last_gripper_centers: dict[int, np.ndarray] = {}
    last_gripper_ts: dict[int, int] = {}

    timestamps = frame_summary_df["timestamp_ms"].astype(int).tolist()
    frame_meta_lookup = {
        int(row["timestamp_ms"]): row
        for _, row in frame_summary_df.iterrows()
    }
    for ts in timestamps:
        group = (
            df[df["timestamp_ms"].eq(ts)].copy()
            if not df.empty
            else pd.DataFrame()
        )
        summary: dict[str, Any] = {
            "timestamp_ms": ts,
            "detected_marker_count": int(len(group)),
        }

        # -------------------------------------------------------------
        # Delivery / Task — downward-ray AOIs
        #
        # Actual layout:
        #   upper row: 5 -- 3 -- 1
        #   lower row: 6 -- 4 -- 2
        #
        # Task:     top 5--3, rays 5->6 and 3->4.
        # Delivery: top 3--1, rays 3->4 and 1->2.
        #
        # 4 visible markers -> rays extend to the image bottom.
        # 3 visible markers -> conservative direct triangle.
        # <3 visible markers -> unavailable.
        #
        # No missing corner is estimated.
        # -------------------------------------------------------------
        frame_meta = frame_meta_lookup.get(ts, {})
        scene_motion_reset = bool(
            frame_meta.get("scene_motion_reset", False)
        )

        table_group = (
            group[group["marker_id"].isin(table_ids)].copy()
            if not group.empty
            else pd.DataFrame()
        )
        direct_table_group = (
            table_group[
                table_group["detection_source"].eq("aruco_direct")
            ].copy()
            if not table_group.empty
            and "detection_source" in table_group.columns
            else pd.DataFrame()
        )

        current_table = marker_centers(table_group, table_ids)
        direct_table = marker_centers(
            direct_table_group,
            table_ids,
        )

        # Retained only as a diagnostic. Polygon construction itself is
        # independent of numeric left/right order.
        table_pair_columns, pair_updates = (
            update_table_pair_orientation(
                direct_table,
                table_pair_columns,
            )
        )

        table_sources = (
            table_group["detection_source"]
            .value_counts()
            .to_dict()
            if not table_group.empty
            and "detection_source" in table_group.columns
            else {}
        )

        region_modes: dict[str, str] = {}

        for region_name, expected_ids in TABLE_REGION_MARKER_IDS.items():
            region_polygon, found_ids, construction_mode = (
                build_direct_region_hull(
                    current_table,
                    expected_ids,
                    region_name=region_name,
                    outer_expansion_px=float(
                        cfg.table_outer_expansion_px
                    ),
                    frame_width=float(
                        cfg.aoi_frame_width or 1920.0
                    ),
                    frame_height=float(
                        cfg.aoi_frame_height or 1080.0
                    ),
                    ray_bottom_margin_px=float(
                        cfg.table_ray_bottom_margin_px
                    ),
                )
            )

            held_ids = []
            if found_ids and not table_group.empty:
                held_ids = sorted(
                    table_group[
                        table_group["marker_id"].isin(found_ids)
                        & table_group["detection_source"].astype(str).eq(
                            "validated_short_hold"
                        )
                    ]["marker_id"]
                    .astype(int)
                    .unique()
                    .tolist()
                )

            if region_polygon is not None and held_ids:
                construction_mode = (
                    f"{construction_mode}_validated_hold"
                )

            available = region_polygon is not None
            region_modes[region_name] = (
                construction_mode if available else "unavailable"
            )

            aoi_rows.append(
                {
                    "timestamp_ms": ts,
                    "aoi_name": region_name,
                    "geometry_type": "polygon",
                    "available": available,
                    "points_json": (
                        to_json_or_empty(
                            region_polygon.tolist()
                        )
                        if available
                        else ""
                    ),
                    "center_json": "",
                    "radius_px": "",
                    "found_marker_ids": ",".join(
                        map(str, found_ids)
                    ),
                    "expected_marker_ids": ",".join(
                        map(str, expected_ids)
                    ),
                    "construction_mode": (
                        construction_mode
                        if available
                        else "unavailable"
                    ),
                    "reference_marker_count": len(found_ids),
                    "used_reference_ids": ",".join(
                        map(str, found_ids)
                    ),
                    "validated_hold_marker_ids": ",".join(
                        map(str, held_ids)
                    ),
                    "note": (
                        "4 markers = quadrilateral; 3 markers = triangle; "
                        "one short missing marker may use a forward-backward "
                        "validated previous position"
                    ),
                }
            )

            summary[f"{region_name}_available"] = available
            summary[f"{region_name}_construction_mode"] = (
                construction_mode
                if available
                else "unavailable"
            )
            summary[f"{region_name}_reference_count"] = len(
                found_ids
            )
            summary[f"{region_name}_reference_ids"] = ",".join(
                map(str, found_ids)
            )
            summary[f"{region_name}_validated_hold_ids"] = ",".join(
                map(str, held_ids)
            )

        summary["Table_reference_count"] = len(current_table)
        summary["Table_reference_ids"] = ",".join(
            map(str, sorted(current_table))
        )
        summary["Table_direct_marker_count"] = len(direct_table)
        summary["Table_direct_marker_ids"] = ",".join(
            map(str, sorted(direct_table))
        )
        summary["Table_flow_marker_count"] = int(
            table_sources.get("optical_flow", 0)
        )
        summary["Table_pair_orientation"] = ";".join(
            f"{marker_id}:{'L' if column == 0 else 'R'}"
            for marker_id, column in sorted(
                table_pair_columns.items()
            )
        )
        summary["Table_pair_orientation_updates"] = ";".join(
            pair_updates
        )
        summary["Table_mapping_mode"] = (
            f"Delivery={region_modes['Delivery']};"
            f"Task={region_modes['Task']}"
        )
        summary["Scene_motion_reset"] = scene_motion_reset
        summary["Scene_motion_median_px"] = frame_meta.get(
            "scene_motion_median_px",
            np.nan,
        )
        summary["Scene_motion_p90_px"] = frame_meta.get(
            "scene_motion_p90_px",
            np.nan,
        )

        # -------------------------------------------------------------
        # RobotArm / corridor
        # - one visible marker: one circle
        # - two, three, or four visible markers: one connected rotated ellipse
        #
        # Marker smoothing and optical-flow rejection remain unchanged.
        # -------------------------------------------------------------
        current_robot = marker_centers(group, robot_ids)
        accepted_robot_centers: dict[int, np.ndarray] = {}
        robot_modes: dict[int, str] = {}
        rejected_robot_ids: list[int] = []

        for marker_id in sorted(current_robot):
            raw_center = current_robot[marker_id]
            source = detection_source_for_marker(group, marker_id)
            previous = last_robot_centers.get(marker_id)
            previous_ts = last_robot_ts.get(marker_id)
            recent = (
                previous is not None
                and previous_ts is not None
                and (ts - previous_ts)
                <= float(cfg.temporal_quality_window_ms)
            )

            if recent:
                jump = float(np.linalg.norm(raw_center - previous))
                if (
                    source == "optical_flow"
                    and jump
                    > float(cfg.optical_flow_max_center_step_px)
                ):
                    rejected_robot_ids.append(marker_id)
                    continue
                center = smooth_points(
                    raw_center,
                    previous,
                    cfg.robot_smoothing_alpha,
                )
                mode = f"{source}_smoothed_{marker_id}"
            else:
                center = raw_center
                mode = f"{source}_reacquired_{marker_id}"

            center = np.asarray(center, dtype=float)
            last_robot_centers[marker_id] = center
            last_robot_ts[marker_id] = ts
            accepted_robot_centers[marker_id] = center
            robot_modes[marker_id] = mode

        robot_found_ids = sorted(accepted_robot_centers)
        robot_available = bool(robot_found_ids)

        if len(robot_found_ids) >= 2:
            ellipse = multi_marker_connected_ellipse_polygon(
                [
                    accepted_robot_centers[marker_id]
                    for marker_id in robot_found_ids
                ],
                float(cfg.robot_arm_marker_radius_px),
                point_count=64,
            )
            marker_count = len(robot_found_ids)
            aoi_rows.append(
                {
                    "timestamp_ms": ts,
                    "aoi_name": "RobotArm",
                    "geometry_type": "polygon",
                    "available": True,
                    "points_json": to_json_or_empty(
                        ellipse.tolist()
                    ),
                    "center_json": "",
                    "radius_px": "",
                    "found_marker_ids": ",".join(
                        map(str, robot_found_ids)
                    ),
                    "expected_marker_ids": "9,10,11,12",
                    "construction_mode": (
                        f"arm_aligned_ellipse_{marker_count}_markers"
                    ),
                    "reference_marker_count": marker_count,
                    "used_reference_ids": ",".join(
                        map(str, robot_found_ids)
                    ),
                    "note": (
                        "single rotated ellipse enclosing all current "
                        "RobotArm/corridor markers"
                    ),
                }
            )
            robot_construction_mode = (
                f"arm_aligned_ellipse_{marker_count}_markers"
            )

        elif len(robot_found_ids) == 1:
            marker_id = robot_found_ids[0]
            aoi_rows.append(
                {
                    "timestamp_ms": ts,
                    "aoi_name": "RobotArm",
                    "geometry_type": "circle",
                    "available": True,
                    "points_json": "",
                    "center_json": to_json_or_empty(
                        accepted_robot_centers[
                            marker_id
                        ].tolist()
                    ),
                    "radius_px": float(
                        cfg.robot_arm_marker_radius_px
                    ),
                    "found_marker_ids": str(marker_id),
                    "expected_marker_ids": "9,10,11,12",
                    "construction_mode": (
                        f"single_circle_{robot_modes[marker_id]}"
                    ),
                    "reference_marker_count": 1,
                    "used_reference_ids": str(marker_id),
                    "note": "single RobotArm/corridor marker circle",
                }
            )
            robot_construction_mode = "single_circle"

        else:
            aoi_rows.append(
                {
                    "timestamp_ms": ts,
                    "aoi_name": "RobotArm",
                    "geometry_type": "circle",
                    "available": False,
                    "points_json": "",
                    "center_json": "",
                    "radius_px": "",
                    "found_marker_ids": "",
                    "expected_marker_ids": "9,10,11,12",
                    "construction_mode": "unavailable",
                    "reference_marker_count": 0,
                    "used_reference_ids": "",
                    "note": (
                        "no current direct/short-flow RobotArm marker"
                    ),
                }
            )
            robot_construction_mode = "unavailable"

        summary["RobotArm_available"] = robot_available
        summary["RobotArm_construction_mode"] = (
            robot_construction_mode
        )
        summary["RobotArm_found_ids"] = ",".join(
            map(str, robot_found_ids)
        )
        summary["RobotArm_rejected_ids"] = ",".join(
            map(str, rejected_robot_ids)
        )
        summary["RobotArm_visible_marker_count"] = len(
            robot_found_ids
        )

        # -------------------------------------------------------------
        # Gripper
        # - one visible marker among IDs 13–15: one oriented ellipse
        # - two or three visible markers: one connected rotated ellipse
        # -------------------------------------------------------------
        current_gripper = marker_centers(group, gripper_ids)
        accepted_gripper_centers: dict[int, np.ndarray] = {}
        gripper_modes: dict[int, str] = {}
        rejected_gripper_ids: list[int] = []

        for marker_id in sorted(current_gripper):
            raw_center = current_gripper[marker_id]
            source = detection_source_for_marker(
                group,
                marker_id,
            )
            previous = last_gripper_centers.get(marker_id)
            previous_ts = last_gripper_ts.get(marker_id)
            recent = (
                previous is not None
                and previous_ts is not None
                and (ts - previous_ts)
                <= float(cfg.temporal_quality_window_ms)
            )

            if recent:
                jump = float(
                    np.linalg.norm(raw_center - previous)
                )
                if (
                    source == "optical_flow"
                    and jump
                    > float(cfg.optical_flow_max_center_step_px)
                ):
                    rejected_gripper_ids.append(marker_id)
                    continue
                center = smooth_points(
                    raw_center,
                    previous,
                    cfg.gripper_smoothing_alpha,
                )
                mode = f"{source}_smoothed_{marker_id}"
            else:
                center = raw_center
                mode = f"{source}_reacquired_{marker_id}"

            center = np.asarray(center, dtype=float)
            last_gripper_centers[marker_id] = center
            last_gripper_ts[marker_id] = ts
            accepted_gripper_centers[marker_id] = center
            gripper_modes[marker_id] = mode

        gripper_found_ids = sorted(
            accepted_gripper_centers
        )
        gripper_available = bool(gripper_found_ids)

        if len(gripper_found_ids) >= 2:
            gripper_axis_angle, gripper_axis_source = (
                gripper_arm_axis_angle_rad(
                    robot_centers=accepted_robot_centers,
                    gripper_centers=accepted_gripper_centers,
                    group_df=group,
                    fallback_marker_id=gripper_found_ids[0],
                    marker_angle_offset_deg=float(
                        cfg.gripper_single_marker_angle_offset_deg
                    ),
                )
            )
            ellipse = arm_aligned_gripper_ellipse_polygon(
                centers=[
                    accepted_gripper_centers[marker_id]
                    for marker_id in gripper_found_ids
                ],
                axis_angle_rad=gripper_axis_angle,
                envelope_radius_px=float(
                    cfg.gripper_single_marker_radius_px
                ),
                minimum_major_radius_px=float(
                    cfg.gripper_single_marker_major_radius_px
                ),
                minimum_minor_radius_px=float(
                    cfg.gripper_single_marker_minor_radius_px
                ),
                point_count=64,
            )
            marker_count = len(gripper_found_ids)
            aoi_rows.append(
                {
                    "timestamp_ms": ts,
                    "aoi_name": "Gripper",
                    "geometry_type": "polygon",
                    "available": True,
                    "points_json": to_json_or_empty(
                        ellipse.tolist()
                    ),
                    "center_json": "",
                    "radius_px": "",
                    "found_marker_ids": ",".join(
                        map(str, gripper_found_ids)
                    ),
                    "expected_marker_ids": "13,14,15",
                    "construction_mode": (
                        f"connected_ellipse_{marker_count}_markers"
                    ),
                    "reference_marker_count": marker_count,
                    "used_reference_ids": ",".join(
                        map(str, gripper_found_ids)
                    ),
                    "gripper_axis_angle_deg": float(
                        np.rad2deg(gripper_axis_angle)
                    ),
                    "gripper_axis_source": gripper_axis_source,
                    "note": (
                        "one expanded ellipse enclosing all visible gripper markers; "
                        "major axis follows the current robot-arm direction"
                    ),
                }
            )
            gripper_construction_mode = (
                f"connected_ellipse_{marker_count}_markers"
            )

        elif len(gripper_found_ids) == 1:
            marker_id = gripper_found_ids[0]
            gripper_axis_angle, gripper_axis_source = (
                gripper_arm_axis_angle_rad(
                    robot_centers=accepted_robot_centers,
                    gripper_centers=accepted_gripper_centers,
                    group_df=group,
                    fallback_marker_id=marker_id,
                    marker_angle_offset_deg=float(
                        cfg.gripper_single_marker_angle_offset_deg
                    ),
                )
            )

            ellipse = arm_aligned_gripper_ellipse_polygon(
                centers=[
                    accepted_gripper_centers[marker_id]
                ],
                axis_angle_rad=gripper_axis_angle,
                envelope_radius_px=0.0,
                minimum_major_radius_px=float(
                    cfg.gripper_single_marker_major_radius_px
                ),
                minimum_minor_radius_px=float(
                    cfg.gripper_single_marker_minor_radius_px
                ),
                point_count=48,
            )

            aoi_rows.append(
                {
                    "timestamp_ms": ts,
                    "aoi_name": "Gripper",
                    "geometry_type": "polygon",
                    "available": True,
                    "points_json": to_json_or_empty(
                        ellipse.tolist()
                    ),
                    "center_json": "",
                    "radius_px": "",
                    "found_marker_ids": str(marker_id),
                    "expected_marker_ids": "13,14,15",
                    "construction_mode": (
                        f"single_arm_aligned_ellipse_{gripper_modes[marker_id]}"
                    ),
                    "reference_marker_count": 1,
                    "used_reference_ids": str(marker_id),
                    "gripper_axis_angle_deg": float(
                        np.rad2deg(gripper_axis_angle)
                    ),
                    "gripper_axis_source": gripper_axis_source,
                    "note": (
                        "single expanded ellipse centred on the visible marker; "
                        "major axis follows the robot-arm direction"
                    ),
                }
            )
            gripper_construction_mode = (
                "single_arm_aligned_ellipse"
            )

        else:
            aoi_rows.append(
                {
                    "timestamp_ms": ts,
                    "aoi_name": "Gripper",
                    "geometry_type": "polygon",
                    "available": False,
                    "points_json": "",
                    "center_json": "",
                    "radius_px": "",
                    "found_marker_ids": "",
                    "expected_marker_ids": "13,14,15",
                    "construction_mode": "unavailable",
                    "reference_marker_count": 0,
                    "used_reference_ids": "",
                    "note": (
                        "no current direct/short-flow gripper marker"
                    ),
                }
            )
            gripper_construction_mode = "unavailable"

        summary["Gripper_available"] = gripper_available
        summary["Gripper_construction_mode"] = (
            gripper_construction_mode
        )
        summary["Gripper_found_ids"] = ",".join(
            map(str, gripper_found_ids)
        )
        summary["Gripper_rejected_ids"] = ",".join(
            map(str, rejected_gripper_ids)
        )
        summary["Gripper_visible_marker_count"] = len(
            gripper_found_ids
        )
        summary["Gripper_axis_angle_deg"] = (
            float(np.rad2deg(gripper_axis_angle))
            if gripper_found_ids
            else np.nan
        )
        summary["Gripper_axis_source"] = (
            gripper_axis_source
            if gripper_found_ids
            else "unavailable"
        )

        summary["RobotTotal_available"] = bool(
            robot_available or gripper_available
        )
        summary["RobotTotal_construction_mode"] = (
            "RobotArm+Gripper"
            if robot_available and gripper_available
            else "RobotArm_only"
            if robot_available
            else "Gripper_only"
            if gripper_available
            else "unavailable"
        )
        summary_rows.append(summary)

    aoi_df = pd.DataFrame(aoi_rows)
    aoi_summary_df = pd.DataFrame(summary_rows)
    availability_df = availability_summary(aoi_summary_df)

    orientation_path = (
        out_dir / f"{stem}_table_pair_orientation_shared6_v6_triangle_robotellipse.json"
    )
    orientation_payload = {
        "method": (
            "automatic image-left/image-right assignment within "
            "row pairs {1,2}, {3,4}, {5,6}"
        ),
        "marker_columns": {
            str(marker_id): (
                "left" if column == 0 else "right"
            )
            for marker_id, column in sorted(
                table_pair_columns.items()
            )
        },
        "note": (
            "Numeric IDs may be physically swapped within a row. "
            "The saved orientation is inferred from direct detections."
        ),
    }
    orientation_path.write_text(
        json.dumps(
            orientation_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  table pair orientation saved: {orientation_path}")

    aoi_csv = out_dir / f"{stem}_dynamic_aoi_shared6_v6_triangle_robotellipse.csv"
    summary_csv = out_dir / f"{stem}_dynamic_aoi_summary_shared6_v6_triangle_robotellipse.csv"
    availability_csv = out_dir / f"{stem}_aoi_availability_shared6_v6_triangle_robotellipse.csv"
    write_csv(aoi_df, aoi_csv)
    write_csv(aoi_summary_df, summary_csv)
    write_csv(availability_df, availability_csv)

    print(f"  saved: {aoi_csv}")
    print(f"  saved: {summary_csv}")
    print(f"  saved: {availability_csv}")
    if not availability_df.empty:
        print("\n  AOI availability:")
        print(availability_df.to_string(index=False))

    if not aoi_summary_df.empty:
        if "Table_reference_count" in aoi_summary_df.columns:
            print("\n  Table reference-count distribution:")
            print(
                aoi_summary_df["Table_reference_count"]
                .fillna(-1)
                .astype(int)
                .value_counts()
                .sort_index()
                .to_string()
            )
        if "Table_mapping_mode" in aoi_summary_df.columns:
            print("\n  Simple table mapping-mode counts:")
            print(
                aoi_summary_df["Table_mapping_mode"]
                .fillna("unavailable")
                .value_counts()
                .to_string()
            )
        if "Table_pair_orientation" in aoi_summary_df.columns:
            orientations = (
                aoi_summary_df["Table_pair_orientation"]
                .dropna()
                .astype(str)
            )
            orientations = orientations[orientations.str.len() > 0]
            if not orientations.empty:
                print(
                    "\n  Final inferred table pair orientation: "
                    + orientations.iloc[-1]
                )

    return (
        aoi_df,
        aoi_summary_df,
        availability_df,
        aoi_csv,
        summary_csv,
        availability_csv,
    )


def choose_column(df: pd.DataFrame, explicit: str | None, candidates: list[str], label: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Specified {label} column not found: {explicit}")
        return explicit
    for col in candidates:
        if col in df.columns:
            return col
    if label == "gaze_x":
        for c in df.columns:
            lc = c.lower()
            if "gaze" in lc and lc.endswith("x"):
                return c
    if label == "gaze_y":
        for c in df.columns:
            lc = c.lower()
            if "gaze" in lc and lc.endswith("y"):
                return c
    if label == "time":
        for c in df.columns:
            if "time" in c.lower():
                return c
    raise ValueError(f"Could not auto-detect {label} column. Available columns:\n" + "\n".join(df.columns))


def parse_time_to_ms(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    if ":" in s:
        parts = s.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return np.nan
        if len(nums) == 3:
            h, m, sec = nums
            return (h * 3600 + m * 60 + sec) * 1000
        if len(nums) == 2:
            m, sec = nums
            return (m * 60 + sec) * 1000
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def polygon_signed_distance_px(
    x: float,
    y: float,
    points_json: str,
) -> float:
    """
    Signed distance to polygon boundary.

    Positive: inside.
    Zero: boundary.
    Negative: outside.
    """
    if pd.isna(points_json) or points_json == "":
        return np.nan
    try:
        points = np.array(
            json.loads(points_json),
            dtype=np.float32,
        )
    except Exception:
        return np.nan
    if points.ndim != 2 or points.shape[0] < 3:
        return np.nan
    try:
        return float(
            cv2.pointPolygonTest(
                points,
                (float(x), float(y)),
                True,
            )
        )
    except cv2.error:
        return np.nan


def point_in_polygon(
    x: float,
    y: float,
    points_json: str,
) -> bool:
    distance = polygon_signed_distance_px(
        x,
        y,
        points_json,
    )
    return bool(
        np.isfinite(distance)
        and distance >= 0.0
    )


def circle_signed_distance_px(
    x: float,
    y: float,
    center_json: str,
    radius_px: Any,
) -> float:
    """
    Signed distance to circle boundary.

    Positive: inside.
    Zero: boundary.
    Negative: outside.
    """
    if (
        pd.isna(center_json)
        or center_json == ""
        or pd.isna(radius_px)
        or radius_px == ""
    ):
        return np.nan
    try:
        center = np.array(
            json.loads(center_json),
            dtype=float,
        )
        radius = float(radius_px)
    except Exception:
        return np.nan

    if center.size != 2 or radius <= 0.0:
        return np.nan

    center_distance = float(
        np.sqrt(
            (float(x) - center[0]) ** 2
            + (float(y) - center[1]) ** 2
        )
    )
    return float(radius - center_distance)


def point_in_circle(
    x: float,
    y: float,
    center_json: str,
    radius_px: Any,
) -> bool:
    distance = circle_signed_distance_px(
        x,
        y,
        center_json,
        radius_px,
    )
    return bool(
        np.isfinite(distance)
        and distance >= 0.0
    )


def geometry_signed_distance_px(
    x: float,
    y: float,
    row: pd.Series,
) -> float:
    geometry_type = str(
        row.get("geometry_type", "")
    ).lower()

    if geometry_type == "polygon":
        return polygon_signed_distance_px(
            x,
            y,
            row.get("points_json", ""),
        )

    if geometry_type == "circle":
        return circle_signed_distance_px(
            x,
            y,
            row.get("center_json", ""),
            row.get("radius_px", ""),
        )

    return np.nan


def build_aoi_lookup(
    dynamic_aoi_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[int, pd.DataFrame]]:
    df = dynamic_aoi_df.copy()
    df["timestamp_ms"] = pd.to_numeric(
        df["timestamp_ms"],
        errors="coerce",
    ).astype("Int64")
    df = df.dropna(
        subset=["timestamp_ms"]
    ).copy()
    df["timestamp_ms"] = df[
        "timestamp_ms"
    ].astype(int)
    df["available"] = (
        df["available"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    timestamps = np.array(
        sorted(
            df["timestamp_ms"]
            .unique()
            .tolist()
        ),
        dtype=int,
    )
    lookup = {
        int(timestamp): df[
            df["timestamp_ms"].eq(timestamp)
        ].copy()
        for timestamp in timestamps
    }
    return timestamps, lookup


def nearest_timestamp(
    t: float,
    timestamps: np.ndarray,
    max_nearest_ms: float,
) -> tuple[int | None, float]:
    if np.isnan(t) or len(timestamps) == 0:
        return None, np.nan
    index = int(
        np.searchsorted(timestamps, t)
    )
    candidates: list[int] = []
    if index < len(timestamps):
        candidates.append(
            int(timestamps[index])
        )
    if index > 0:
        candidates.append(
            int(timestamps[index - 1])
        )
    if not candidates:
        return None, np.nan

    best = min(
        candidates,
        key=lambda value: abs(
            float(value) - float(t)
        ),
    )
    difference = abs(
        float(best) - float(t)
    )
    if difference <= max_nearest_ms:
        return int(best), difference
    return None, difference


def classify_one_gaze_detailed(
    x: float,
    y: float,
    aoi_rows: pd.DataFrame,
    robot_near_enabled: bool,
    gripper_margin_px: float,
    robot_arm_margin_px: float,
) -> tuple[
    str,
    str,
    bool,
    bool,
    str,
    float,
]:
    """
    Return:
      final_label,
      strict_label,
      any_aoi_available,
      robot_near_flag,
      robot_near_source,
      robot_near_distance_px

    Direct AOI priority remains:
      Gripper > RobotArm > Delivery > Task.

    RobotNear is evaluated only if no direct AOI contains the gaze. Therefore
    it can only replace Outside and cannot steal gaze from Task or Delivery.
    """
    available_rows = aoi_rows[
        aoi_rows["available"]
    ].copy()

    if available_rows.empty:
        return (
            "NoAOIData",
            "NoAOIData",
            False,
            False,
            "",
            np.nan,
        )

    # Strict direct classification first.
    for aoi_name in AOI_PRIORITY:
        subset = available_rows[
            available_rows["aoi_name"].eq(
                aoi_name
            )
        ]
        if subset.empty:
            continue

        for _, aoi_row in subset.iterrows():
            geometry_type = str(
                aoi_row.get(
                    "geometry_type",
                    "",
                )
            ).lower()

            inside = (
                geometry_type == "polygon"
                and point_in_polygon(
                    x,
                    y,
                    aoi_row.get(
                        "points_json",
                        "",
                    ),
                )
            ) or (
                geometry_type == "circle"
                and point_in_circle(
                    x,
                    y,
                    aoi_row.get(
                        "center_json",
                        "",
                    ),
                    aoi_row.get(
                        "radius_px",
                        "",
                    ),
                )
            )

            if inside:
                return (
                    aoi_name,
                    aoi_name,
                    True,
                    False,
                    "",
                    0.0,
                )

    strict_label = "Outside"

    if not bool(robot_near_enabled):
        return (
            strict_label,
            strict_label,
            True,
            False,
            "",
            np.nan,
        )

    best_source = ""
    best_distance = np.inf

    for robot_aoi, margin in [
        ("Gripper", float(gripper_margin_px)),
        ("RobotArm", float(robot_arm_margin_px)),
    ]:
        if margin <= 0.0:
            continue

        subset = available_rows[
            available_rows["aoi_name"].eq(
                robot_aoi
            )
        ]
        for _, aoi_row in subset.iterrows():
            signed_distance = (
                geometry_signed_distance_px(
                    x,
                    y,
                    aoi_row,
                )
            )
            if not np.isfinite(signed_distance):
                continue

            outside_distance = max(
                0.0,
                -float(signed_distance),
            )

            if (
                outside_distance <= margin
                and outside_distance
                < best_distance
            ):
                best_distance = outside_distance
                best_source = robot_aoi

    if best_source:
        return (
            ROBOT_NEAR_LABEL,
            strict_label,
            True,
            True,
            best_source,
            float(best_distance),
        )

    return (
        strict_label,
        strict_label,
        True,
        False,
        "",
        np.nan,
    )


def classify_one_gaze(
    x: float,
    y: float,
    aoi_rows: pd.DataFrame,
) -> tuple[str, bool]:
    """
    Backward-compatible strict classifier.
    """
    (
        _final_label,
        strict_label,
        aoi_available,
        _near_flag,
        _near_source,
        _near_distance,
    ) = classify_one_gaze_detailed(
        x=x,
        y=y,
        aoi_rows=aoi_rows,
        robot_near_enabled=False,
        gripper_margin_px=0.0,
        robot_arm_margin_px=0.0,
    )
    return strict_label, aoi_available


def compute_dt_ms(times: pd.Series) -> pd.Series:
    t = pd.to_numeric(times, errors="coerce")
    diff_next = t.shift(-1) - t
    positive = diff_next[(diff_next > 0) & (diff_next < 200)]
    median_dt = float(positive.median()) if len(positive) else 16.7
    dt = diff_next.copy()
    dt[(dt <= 0) | (dt.isna()) | (dt > 200)] = median_dt
    return dt.fillna(median_dt).clip(lower=0, upper=200)



def assign_valid_stream_durations(
    samples: pd.DataFrame,
    continuity_gap_ms: float,
) -> pd.Series:
    """
    Assign analytical duration only to unique valid gaze samples.

    D-Lab CsvData exports may interleave rows from multiple streams. Empty gaze
    rows therefore must not consume time. A valid gaze sample covers the
    interval to the next valid sample when that gap is within the continuity
    threshold; otherwise it covers one expected sample interval and the
    remainder is treated as tracking loss.
    """
    durations = pd.Series(0.0, index=samples.index, dtype=float)
    valid = samples.loc[
        samples["valid_gaze"] & samples["gaze_time_ms"].notna(),
        ["gaze_time_ms"],
    ].copy()
    if valid.empty:
        return durations

    valid["gaze_time_ms"] = pd.to_numeric(
        valid["gaze_time_ms"], errors="coerce"
    )
    valid = valid.dropna().sort_values("gaze_time_ms")
    # One analytical sample per timestamp; duplicate export rows receive 0 ms.
    valid = valid[~valid["gaze_time_ms"].duplicated(keep="first")]

    times = valid["gaze_time_ms"].to_numpy(dtype=float)
    if len(times) >= 2:
        diffs = np.diff(times)
        plausible = diffs[
            (diffs > 0)
            & (diffs <= float(continuity_gap_ms))
        ]
        expected_interval = (
            float(np.median(plausible))
            if len(plausible)
            else 16.7
        )
    else:
        expected_interval = 16.7

    expected_interval = max(0.1, expected_interval)
    assigned = np.full(len(times), expected_interval, dtype=float)
    if len(times) >= 2:
        gaps = np.diff(times)
        assigned[:-1] = np.where(
            gaps <= float(continuity_gap_ms),
            gaps,
            expected_interval,
        )
    durations.loc[valid.index] = assigned
    return durations



def _raw_blank_mask(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return series.isna() | text.isin({"", "nan", "none", "null", "na", "n/a"})


def detect_eye_pupil_source_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Detect raw left/right pupil columns used only for quality diagnostics.

    These row-level availability ratios are descriptive. They are not
    interpreted as eye-tracker sampling-rate estimates because D-Lab exports
    may interleave multiple data streams.
    """
    result = {"left": [], "right": []}
    for column in df.columns:
        lowered = str(column).lower()
        if "pupil" not in lowered:
            continue
        if "left eye" in lowered:
            result["left"].append(column)
        elif "right eye" in lowered:
            result["right"].append(column)
    return result


def any_positive_numeric(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(False, index=df.index, dtype=bool)
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.gt(0).any(axis=1)


def assign_gaze_quality_status(
    samples: pd.DataFrame,
    gaze_x_raw_text: pd.Series,
    gaze_y_raw_text: pd.Series,
    gaze_native_width: float,
    gaze_native_height: float,
    aoi_frame_width: float,
    aoi_frame_height: float,
) -> pd.Series:
    """
    Split the old mixed InvalidGaze category into explicit row-level reasons.

    This status is a raw-export diagnostic. Real tracking loss is estimated
    separately from gaps in the valid scene-gaze stream.
    """
    x_blank = _raw_blank_mask(gaze_x_raw_text)
    y_blank = _raw_blank_mask(gaze_y_raw_text)
    x_num = samples["gaze_x_raw"]
    y_num = samples["gaze_y_raw"]

    status = pd.Series("ValidGaze", index=samples.index, dtype="object")

    missing_time = ~samples["time_valid"]
    missing_xy = ~missing_time & (x_blank | y_blank)
    non_numeric = (
        ~missing_time
        & ~missing_xy
        & (x_num.isna() | y_num.isna())
    )
    zero_or_negative = (
        ~missing_time
        & ~missing_xy
        & ~non_numeric
        & ((x_num <= 0) | (y_num <= 0))
    )
    outside_native = (
        ~missing_time
        & ~missing_xy
        & ~non_numeric
        & ~zero_or_negative
        & ((x_num > gaze_native_width) | (y_num > gaze_native_height))
    )
    outside_aoi_frame = (
        ~missing_time
        & ~missing_xy
        & ~non_numeric
        & ~zero_or_negative
        & ~outside_native
        & (
            (samples["gaze_x"] <= 0)
            | (samples["gaze_y"] <= 0)
            | (samples["gaze_x"] > aoi_frame_width)
            | (samples["gaze_y"] > aoi_frame_height)
        )
    )

    status.loc[missing_time] = "MissingTime"
    status.loc[missing_xy] = "MissingGazeXY"
    status.loc[non_numeric] = "NonNumericGazeXY"
    status.loc[zero_or_negative] = "ZeroOrNegativeGaze"
    status.loc[outside_native] = "OutOfNativeFrame"
    status.loc[outside_aoi_frame] = "OutOfAOIFrame"
    return status


def estimate_gaze_stream_quality(
    samples: pd.DataFrame,
    continuity_gap_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Estimate real scene-gaze coverage from temporal gaps between valid gaze
    samples instead of counting every empty export row as tracking loss.

    A gap no larger than continuity_gap_ms is treated as continuous gaze
    coverage. For a larger gap, one expected sample interval is retained and
    the remainder is treated as an estimated tracking-loss run.
    """
    parsed_times = (
        pd.to_numeric(samples.loc[samples["time_valid"], "gaze_time_ms"], errors="coerce")
        .dropna()
        .sort_values()
    )
    valid_times = (
        pd.to_numeric(samples.loc[samples["valid_gaze"], "gaze_time_ms"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype=float)
    )

    if parsed_times.empty:
        summary = {
            "recording_start_ms": float("nan"),
            "recording_end_ms": float("nan"),
            "recording_duration_ms": 0.0,
            "valid_gaze_unique_sample_count": 0,
            "expected_gaze_interval_ms": float("nan"),
            "continuity_gap_threshold_ms": float(continuity_gap_ms),
            "estimated_valid_gaze_coverage_ms": 0.0,
            "estimated_valid_gaze_coverage_ratio": float("nan"),
            "estimated_tracking_loss_ms": 0.0,
            "estimated_tracking_loss_ratio": float("nan"),
            "longest_gap_between_valid_gaze_ms": float("nan"),
            "longest_estimated_tracking_loss_run_ms": float("nan"),
            "tracking_loss_run_count": 0,
            "tracking_loss_run_count_over_500ms": 0,
            "tracking_loss_run_count_over_1000ms": 0,
        }
        return pd.DataFrame([summary]), pd.DataFrame(), summary

    recording_start = float(parsed_times.min())
    recording_end = float(parsed_times.max())

    if len(valid_times) >= 2:
        diffs = np.diff(valid_times)
        plausible = diffs[(diffs > 0) & (diffs <= continuity_gap_ms)]
        if len(plausible):
            expected_interval = float(np.median(plausible))
        else:
            positive = diffs[diffs > 0]
            expected_interval = float(np.median(positive)) if len(positive) else 16.7
    else:
        expected_interval = 16.7

    expected_interval = max(0.1, expected_interval)
    recording_duration = max(
        expected_interval,
        recording_end - recording_start + expected_interval,
    )

    loss_runs: list[dict[str, float | str]] = []
    longest_raw_gap = float("nan")

    if len(valid_times) == 0:
        loss_runs.append({
            "run_type": "NoValidGaze",
            "start_ms": recording_start,
            "end_ms": recording_end + expected_interval,
            "duration_ms": recording_duration,
            "raw_valid_sample_gap_ms": float("nan"),
        })
    else:
        start_loss = max(0.0, float(valid_times[0]) - recording_start)
        if start_loss > 0:
            loss_runs.append({
                "run_type": "RecordingStartToFirstValidGaze",
                "start_ms": recording_start,
                "end_ms": float(valid_times[0]),
                "duration_ms": start_loss,
                "raw_valid_sample_gap_ms": start_loss,
            })

        if len(valid_times) >= 2:
            gaps = np.diff(valid_times)
            longest_raw_gap = float(np.max(gaps))
            for current_t, next_t, raw_gap in zip(valid_times[:-1], valid_times[1:], gaps):
                if raw_gap > continuity_gap_ms:
                    loss_duration = max(0.0, float(raw_gap) - expected_interval)
                    if loss_duration > 0:
                        loss_runs.append({
                            "run_type": "GapBetweenValidGazeSamples",
                            "start_ms": float(current_t + expected_interval),
                            "end_ms": float(next_t),
                            "duration_ms": loss_duration,
                            "raw_valid_sample_gap_ms": float(raw_gap),
                        })

        end_loss = max(0.0, recording_end - float(valid_times[-1]))
        if end_loss > 0:
            loss_runs.append({
                "run_type": "LastValidGazeToRecordingEnd",
                "start_ms": float(valid_times[-1] + expected_interval),
                "end_ms": recording_end + expected_interval,
                "duration_ms": end_loss,
                "raw_valid_sample_gap_ms": end_loss,
            })

    gaps_df = pd.DataFrame(loss_runs)
    estimated_loss = (
        float(gaps_df["duration_ms"].sum())
        if not gaps_df.empty
        else 0.0
    )
    estimated_loss = min(max(estimated_loss, 0.0), recording_duration)
    estimated_valid = max(0.0, recording_duration - estimated_loss)

    longest_loss = (
        float(gaps_df["duration_ms"].max())
        if not gaps_df.empty
        else 0.0
    )
    run_count = int(len(gaps_df))

    summary = {
        "recording_start_ms": recording_start,
        "recording_end_ms": recording_end,
        "recording_duration_ms": recording_duration,
        "valid_gaze_unique_sample_count": int(len(valid_times)),
        "expected_gaze_interval_ms": expected_interval,
        "continuity_gap_threshold_ms": float(continuity_gap_ms),
        "estimated_valid_gaze_coverage_ms": estimated_valid,
        "estimated_valid_gaze_coverage_ratio": (
            estimated_valid / recording_duration if recording_duration else float("nan")
        ),
        "estimated_tracking_loss_ms": estimated_loss,
        "estimated_tracking_loss_ratio": (
            estimated_loss / recording_duration if recording_duration else float("nan")
        ),
        "longest_gap_between_valid_gaze_ms": longest_raw_gap,
        "longest_estimated_tracking_loss_run_ms": longest_loss,
        "tracking_loss_run_count": run_count,
        "tracking_loss_run_count_over_500ms": (
            int((gaps_df["duration_ms"] >= 500).sum()) if not gaps_df.empty else 0
        ),
        "tracking_loss_run_count_over_1000ms": (
            int((gaps_df["duration_ms"] >= 1000).sum()) if not gaps_df.empty else 0
        ),
    }
    return pd.DataFrame([summary]), gaps_df, summary


def build_row_quality_summary(samples: pd.DataFrame) -> pd.DataFrame:
    """
    Row-level export summary.

    Important: ratios here describe D-Lab export rows and must not be reported
    as the instrument's tracking-loss ratio.
    """
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "gaze_quality_status",
                "row_count",
                "raw_row_duration_ms",
                "ratio_of_export_rows",
                "ratio_of_raw_row_duration",
                "interpretation",
            ]
        )

    total_rows = len(samples)
    total_raw_duration = float(samples["raw_row_dt_ms"].sum())

    counts = (
        samples["gaze_quality_status"]
        .value_counts(dropna=False)
        .rename_axis("gaze_quality_status")
        .reset_index(name="row_count")
    )
    durations = (
        samples.groupby("gaze_quality_status", dropna=False)["raw_row_dt_ms"]
        .sum()
        .reset_index(name="raw_row_duration_ms")
    )
    out = counts.merge(durations, on="gaze_quality_status", how="outer")
    out["ratio_of_export_rows"] = out["row_count"] / total_rows if total_rows else np.nan
    out["ratio_of_raw_row_duration"] = (
        out["raw_row_duration_ms"] / total_raw_duration
        if total_raw_duration
        else np.nan
    )
    out["interpretation"] = (
        "D-Lab export-row diagnostic only; do not interpret as tracking loss"
    )
    return out


def detect_pupil_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = list(df.columns)
    return {
        "pupil_area": [c for c in cols if "Pupil Area" in c],
        "pupil_height": [c for c in cols if "Pupil Height" in c],
        "pupil_width": [c for c in cols if "Pupil Width" in c],
    }


def row_mean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)



def normalize_column_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def peek_triggered_tasks_header(path: Path) -> bool:
    """Fast header-only check used when auto-searching data_raw."""
    try:
        raw = Path(path).read_bytes()[:32768]
    except Exception:
        return False

    for encoding in ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030", "latin1"]:
        try:
            decoded = raw.decode(encoding)
        except Exception:
            continue
        lines = [line.strip() for line in decoded.splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0]
        for separator in ["\t", ",", ";"]:
            tokens = {normalize_column_token(x) for x in header.split(separator)}
            has_task = bool(tokens.intersection({"task", "taskname"}))
            has_start = bool(tokens.intersection({"starttime", "start", "startms", "startrecms"}))
            has_end = bool(tokens.intersection({"endtime", "end", "endms", "endrecms"}))
            if has_task and has_start and has_end:
                return True
    return False


def find_triggered_tasks_export(
    gaze_path: Path,
    video_path: Path,
    explicit_path: Path | None,
    auto_find: bool,
) -> Path | None:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Triggered Tasks file not found: {path}")
        return path

    if not auto_find:
        return None

    search_dirs: list[Path] = []
    for directory in [Path(gaze_path).parent, PROJECT_ROOT / "data_raw"]:
        directory = directory.resolve()
        if directory.exists() and directory not in search_dirs:
            search_dirs.append(directory)

    candidates: list[Path] = []
    gaze_resolved = Path(gaze_path).resolve()
    for directory in search_dirs:
        for pattern in ["*.txt", "*.csv", "*.tsv"]:
            for path in directory.glob(pattern):
                if path.resolve() == gaze_resolved:
                    continue
                if peek_triggered_tasks_header(path):
                    candidates.append(path.resolve())

    candidates = sorted(set(candidates))
    if not candidates:
        return None
    if len(candidates) == 1:
        print(f"  Auto-detected Triggered Tasks file: {candidates[0]}")
        return candidates[0]

    def stem_tokens(path: Path) -> set[str]:
        generic = {
            "triggered", "task", "tasks", "export", "data", "raw",
            "csvdata", "gaze", "video", "scene", "cam", "field",
        }
        tokens = set(re.findall(r"[a-z0-9]+", path.stem.lower()))
        return tokens - generic

    reference = stem_tokens(Path(gaze_path)) | stem_tokens(Path(video_path))
    scored = [(len(stem_tokens(path) & reference), path) for path in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)

    if scored and scored[0][0] > 0:
        top_score = scored[0][0]
        top = [path for score, path in scored if score == top_score]
        if len(top) == 1:
            print(f"  Auto-detected Triggered Tasks file by filename match: {top[0]}")
            return top[0]

    listing = "\n".join(f"  - {path}" for path in candidates)
    raise ValueError(
        "Multiple Triggered Tasks files were found. Specify the correct one with "
        "--triggered_tasks_txt.\n" + listing
    )



TIMELINE_BASE_COLUMNS = [
    "user_id",
    "trial_index",
    "task_id",
]

FINE_PHASE_SPECS = [
    ("T1_Motion", "t1_m_start_unix_ns", "t1_m_end_unix_ns"),
    ("T1_SafetyAssessment", "t1_s_start_unix_ns", "t1_s_end_unix_ns"),
    ("T1_Placement", "t1_p_start_unix_ns", "t1_p_end_unix_ns"),
    ("T2_Motion", "t2_m_start_unix_ns", "t2_m_end_unix_ns"),
    ("T2_SafetyAssessment", "t2_s_start_unix_ns", "t2_s_end_unix_ns"),
    ("T2_Placement", "t2_p_start_unix_ns", "t2_p_end_unix_ns"),
]

FINE_TIMELINE_REQUIRED_COLUMNS = (
    TIMELINE_BASE_COLUMNS
    + [
        start_column
        for _phase, start_column, _end_column in FINE_PHASE_SPECS
    ]
    + [
        end_column
        for _phase, _start_column, end_column in FINE_PHASE_SPECS
    ]
)

COARSE_TIMELINE_REQUIRED_COLUMNS = [
    "user_id",
    "trial_index",
    "task_id",
    "t0_start_unix_ns",
    "t0_end_unix_ns",
    "t1_start_unix_ns",
    "t1_end_unix_ns",
    "t2_start_unix_ns",
    "t2_end_unix_ns",
]


def _table_matrix_from_numbers(table: Any) -> list[list[Any]]:
    return [
        [table.cell(row, col).value for col in range(table.num_cols)]
        for row in range(table.num_rows)
    ]


def _schema_match_score(columns: list[str]) -> tuple[int, str]:
    tokens = {
        normalize_column_token(column)
        for column in columns
    }
    fine_tokens = {
        normalize_column_token(column)
        for column in FINE_TIMELINE_REQUIRED_COLUMNS
    }
    coarse_tokens = {
        normalize_column_token(column)
        for column in COARSE_TIMELINE_REQUIRED_COLUMNS
    }

    fine_match = len(tokens.intersection(fine_tokens))
    coarse_match = len(tokens.intersection(coarse_tokens))

    if fine_match >= coarse_match:
        return fine_match, "fine"
    return coarse_match, "coarse"


def _normalize_timeline_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    out.columns = [
        str(column).strip()
        for column in out.columns
    ]

    normalized_to_original = {
        normalize_column_token(column): column
        for column in out.columns
    }

    all_known_columns = sorted(
        set(
            FINE_TIMELINE_REQUIRED_COLUMNS
            + COARSE_TIMELINE_REQUIRED_COLUMNS
            + [
                "block_press_count",
                "warning_note",
                "user_csv_file",
                "robot_log_file",
                "match_note",
                "task_change_unix_ns",
                "task_change_local",
            ]
            + [
                column.replace("_unix_ns", "_local")
                for column in (
                    FINE_TIMELINE_REQUIRED_COLUMNS
                    + COARSE_TIMELINE_REQUIRED_COLUMNS
                )
                if column.endswith("_unix_ns")
            ]
        )
    )

    rename_map: dict[str, str] = {}
    for expected in all_known_columns:
        token = normalize_column_token(expected)
        if token in normalized_to_original:
            rename_map[
                normalized_to_original[token]
            ] = expected

    out = out.rename(columns=rename_map)

    fine_present = all(
        column in out.columns
        for column in FINE_TIMELINE_REQUIRED_COLUMNS
    )
    coarse_present = all(
        column in out.columns
        for column in COARSE_TIMELINE_REQUIRED_COLUMNS
    )

    if fine_present:
        schema = "fine"
        required = FINE_TIMELINE_REQUIRED_COLUMNS
    elif coarse_present:
        schema = "coarse"
        required = COARSE_TIMELINE_REQUIRED_COLUMNS
    else:
        fine_missing = [
            column
            for column in FINE_TIMELINE_REQUIRED_COLUMNS
            if column not in out.columns
        ]
        coarse_missing = [
            column
            for column in COARSE_TIMELINE_REQUIRED_COLUMNS
            if column not in out.columns
        ]
        raise ValueError(
            "Timeline file does not match the fine-phase TSV schema "
            "or the older coarse timeline schema.\n"
            "Missing fine-phase columns: "
            + ", ".join(fine_missing)
            + "\nMissing coarse columns: "
            + ", ".join(coarse_missing)
        )

    for column in TIMELINE_BASE_COLUMNS:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        ).astype("Int64")

    timestamp_recovered_from_local = 0
    timestamp_unrecoverable = 0

    def parse_precise_ns_value(
        raw_value: Any,
        local_value: Any,
    ) -> Any:
        nonlocal timestamp_recovered_from_local
        nonlocal timestamp_unrecoverable

        raw_token = (
            ""
            if raw_value is None or pd.isna(raw_value)
            else str(raw_value).strip()
        )
        local_token = (
            ""
            if local_value is None or pd.isna(local_value)
            else str(local_value).strip()
        )

        missing_tokens = {
            "",
            "nan",
            "None",
            "<NA>",
            "NaT",
        }

        if raw_token in missing_tokens:
            raw_token = ""
        if local_token in missing_tokens:
            local_token = ""

        # A full integer string is safe. Scientific notation is not
        # converted through float because that loses nanosecond precision.
        if re.fullmatch(r"[+-]?\d{17,20}", raw_token):
            try:
                return int(raw_token)
            except Exception:
                pass

        # Recover exact nanoseconds from the paired timezone-aware local
        # timestamp whenever the raw cell is scientific notation or was
        # parsed as a floating-point value.
        if local_token:
            try:
                parsed = pd.Timestamp(local_token)
                timestamp_recovered_from_local += 1
                return int(parsed.value)
            except Exception:
                pass

        if raw_token:
            timestamp_unrecoverable += 1
        return pd.NA

    ns_columns = [
        column
        for column in out.columns
        if column.endswith("_unix_ns")
    ]
    for column in ns_columns:
        local_column = column.replace(
            "_unix_ns",
            "_local",
        )

        if local_column in out.columns:
            local_values = out[local_column]
        else:
            local_values = pd.Series(
                [pd.NA] * len(out),
                index=out.index,
            )

        out[column] = pd.Series(
            [
                parse_precise_ns_value(
                    raw_value,
                    local_value,
                )
                for raw_value, local_value in zip(
                    out[column],
                    local_values,
                )
            ],
            index=out.index,
            dtype="Int64",
        )

    if "block_press_count" in out.columns:
        out["block_press_count"] = pd.to_numeric(
            out["block_press_count"],
            errors="coerce",
        ).astype("Int64")

    for text_column in [
        "warning_note",
        "user_csv_file",
        "robot_log_file",
        "match_note",
    ]:
        if text_column in out.columns:
            out[text_column] = (
                out[text_column]
                .fillna("")
                .astype(str)
            )

    out = out.dropna(
        subset=TIMELINE_BASE_COLUMNS
    ).copy()

    out.attrs["timeline_schema"] = schema
    out.attrs["required_columns"] = required
    out.attrs["timestamp_recovered_from_local"] = int(
        timestamp_recovered_from_local
    )
    out.attrs["timestamp_unrecoverable"] = int(
        timestamp_unrecoverable
    )
    return out, schema


def read_timeline_file(
    path: Path,
    participant_id: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Read the latest fine-phase timeline TSV.

    Supported formats:
      .tsv / .txt / .csv
      .xlsx / .xls
      .numbers (backward compatibility)

    The latest fine schema contains explicit T1/T2 Motion, Safety Assessment,
    and Placement start/end Unix-nanosecond columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Timeline file not found: {path}"
        )

    suffix = path.suffix.lower()
    selected_sheet = ""
    selected_table = ""

    if suffix == ".numbers":
        try:
            from numbers_parser import Document
        except ImportError as exc:
            raise ImportError(
                "Reading .numbers requires numbers-parser. "
                "For the new workflow, using timeline.tsv is preferred."
            ) from exc

        document = Document(str(path))
        candidates: list[
            tuple[int, str, str, pd.DataFrame]
        ] = []

        for sheet in document.sheets:
            for table in sheet.tables:
                matrix = _table_matrix_from_numbers(table)
                if not matrix:
                    continue

                headers = [
                    str(value).strip()
                    if value is not None
                    else ""
                    for value in matrix[0]
                ]
                if not any(headers):
                    continue

                width = len(headers)
                body = [
                    list(row[:width])
                    + [None] * max(0, width - len(row))
                    for row in matrix[1:]
                ]
                candidate_df = pd.DataFrame(
                    [row[:width] for row in body],
                    columns=headers,
                )
                score, _schema = _schema_match_score(
                    candidate_df.columns.tolist()
                )
                candidates.append(
                    (
                        score * 1000 + len(candidate_df),
                        str(sheet.name),
                        str(table.name),
                        candidate_df,
                    )
                )

        if not candidates:
            raise ValueError(
                f"No readable tables found in: {path}"
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )
        (
            _score,
            selected_sheet,
            selected_table,
            raw_df,
        ) = candidates[0]

    elif suffix in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        candidates: list[
            tuple[int, str, pd.DataFrame]
        ] = []

        for sheet_name in workbook.sheet_names:
            candidate_df = pd.read_excel(
                path,
                sheet_name=sheet_name,
            )
            score, _schema = _schema_match_score(
                candidate_df.columns.tolist()
            )
            candidates.append(
                (
                    score * 1000 + len(candidate_df),
                    sheet_name,
                    candidate_df,
                )
            )

        if not candidates:
            raise ValueError(
                f"No readable sheets found in: {path}"
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )
        _score, selected_sheet, raw_df = candidates[0]
        selected_table = "worksheet"

    elif suffix in {
        ".csv",
        ".txt",
        ".tsv",
    }:
        raw_df = read_table_auto(path)
        selected_sheet = "text"
        selected_table = "text"
    else:
        raise ValueError(
            f"Unsupported timeline format: {path.suffix}. "
            "Use .tsv, .txt, .csv, .xlsx, or .numbers."
        )

    normalized, schema = (
        _normalize_timeline_dataframe(raw_df)
    )

    available_ids = sorted(
        normalized["user_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    timeline = normalized[
        normalized["user_id"]
        .astype(int)
        .eq(int(participant_id))
    ].copy()

    if timeline.empty:
        raise ValueError(
            f"No rows found for participant user_id="
            f"{participant_id}. Available user IDs: "
            f"{available_ids}"
        )

    timeline = timeline.sort_values(
        ["trial_index", "task_id"]
    ).reset_index(drop=True)
    timeline.attrs["timeline_schema"] = schema

    duplicated_trials = (
        timeline["trial_index"]
        .duplicated(keep=False)
    )
    if duplicated_trials.any():
        duplicates = (
            timeline.loc[
                duplicated_trials,
                "trial_index",
            ]
            .astype(int)
            .tolist()
        )
        raise ValueError(
            "Duplicate trial_index values: "
            + ", ".join(map(str, duplicates))
        )

    metadata = {
        "timeline_file": str(path),
        "timeline_sheet": selected_sheet,
        "timeline_table": selected_table,
        "timeline_schema": schema,
        "participant_id": int(participant_id),
        "trial_count": int(len(timeline)),
        "timestamp_recovered_from_local": int(
            normalized.attrs.get(
                "timestamp_recovered_from_local",
                0,
            )
        ),
        "timestamp_unrecoverable": int(
            normalized.attrs.get(
                "timestamp_unrecoverable",
                0,
            )
        ),
        "task_ids": ",".join(
            map(
                str,
                timeline["task_id"]
                .astype(int)
                .tolist(),
            )
        ),
    }

    print("\n  Timeline parser:")
    print(f"    file        = {path}")
    print(f"    format      = {suffix}")
    print(f"    schema      = {schema}")
    print(f"    sheet       = {selected_sheet}")
    print(f"    table       = {selected_table}")
    print(f"    participant = {participant_id}")
    print(f"    trials      = {len(timeline)}")
    print(
        "    ns recovered= "
        + str(
            normalized.attrs.get(
                "timestamp_recovered_from_local",
                0,
            )
        )
    )
    print(
        "    unrecoverable= "
        + str(
            normalized.attrs.get(
                "timestamp_unrecoverable",
                0,
            )
        )
    )

    return timeline, metadata


def _correct_timeline_clock_ms(
    source_utc_ms: float,
    anchor_utc_ms: float,
    offset_ms: float,
    drift_ppm: float,
) -> float:
    scale = 1.0 + float(drift_ppm) * 1e-6
    return (
        float(anchor_utc_ms)
        + (float(source_utc_ms) - float(anchor_utc_ms)) * scale
        + float(offset_ms)
    )


def convert_timeline_to_toi(
    timeline: pd.DataFrame,
    alignment: dict[str, Any],
    video_props: dict[str, float],
    cfg: PipelineConfig,
    out_dir: Path,
) -> tuple[pd.DataFrame, Path, Path]:
    """
    Convert external-computer Unix-nanosecond timestamps to DG3 video time.

    Fine-phase TSV output:
      Trial_Overall
      T1_Reference
      T1_Motion
      T1_SafetyAssessment
      T1_Placement
      T2_Manipulated
      T2_Motion
      T2_SafetyAssessment
      T2_Placement

    Incomplete phases are omitted rather than fabricated. They are recorded
    in the alignment QA file.
    """
    origin_utc_ms = pd.to_numeric(
        pd.Series(
            [
                alignment.get(
                    "recording_origin_utc_ms"
                )
            ]
        ),
        errors="coerce",
    ).iloc[0]
    video_start_rec_ms = pd.to_numeric(
        pd.Series(
            [
                alignment.get(
                    "video_start_rec_ms"
                )
            ]
        ),
        errors="coerce",
    ).iloc[0]

    if pd.isna(origin_utc_ms):
        raise ValueError(
            "The gaze file has no usable UTC-to-rec_time "
            "mapping. The external timeline cannot be aligned."
        )
    if pd.isna(video_start_rec_ms):
        video_start_rec_ms = 0.0

    video_duration_ms = (
        float(video_props["frame_count"])
        / float(video_props["fps"])
        * 1000.0
        if float(video_props.get("fps", 0.0)) > 0
        else np.nan
    )

    schema = str(
        timeline.attrs.get(
            "timeline_schema",
            "fine",
        )
    )

    all_ns_columns = [
        column
        for column in timeline.columns
        if column.endswith("_unix_ns")
    ]
    available_ns_values = [
        int(value)
        for column in all_ns_columns
        for value in timeline[column]
        .dropna()
        .tolist()
    ]
    if not available_ns_values:
        raise ValueError(
            "Timeline contains no usable Unix-nanosecond "
            "timestamps."
        )

    anchor_utc_ms = min(
        available_ns_values
    ) / 1e6

    if schema == "fine":
        explicit_phase_specs = FINE_PHASE_SPECS
    else:
        explicit_phase_specs = [
            (
                "T0_Baseline",
                "t0_start_unix_ns",
                "t0_end_unix_ns",
            ),
            (
                "T1_Reference",
                "t1_start_unix_ns",
                "t1_end_unix_ns",
            ),
            (
                "T2_Manipulated",
                "t2_start_unix_ns",
                "t2_end_unix_ns",
            ),
        ]

    rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []

    def convert_ns(
        value: Any,
    ) -> tuple[float, float, float]:
        if value is None or pd.isna(value):
            return np.nan, np.nan, np.nan

        source_utc_ms = int(value) / 1e6
        corrected_utc_ms = _correct_timeline_clock_ms(
            source_utc_ms=source_utc_ms,
            anchor_utc_ms=anchor_utc_ms,
            offset_ms=cfg.timeline_clock_offset_ms,
            drift_ppm=cfg.timeline_clock_drift_ppm,
        )
        rec_ms = (
            corrected_utc_ms
            - float(origin_utc_ms)
        )
        video_ms = (
            rec_ms
            - float(video_start_rec_ms)
        )
        return (
            corrected_utc_ms,
            rec_ms,
            video_ms,
        )

    for _, trial in timeline.iterrows():
        trial_index = int(
            trial["trial_index"]
        )
        task_id = int(trial["task_id"])
        trial_id = (
            f"Trial{trial_index:02d}_"
            f"Task{task_id:02d}"
        )

        warning_note = str(
            trial.get(
                "warning_note",
                "",
            )
            or ""
        ).strip()
        block_press_count = trial.get(
            "block_press_count",
            pd.NA,
        )

        explicit_rows: list[
            dict[str, Any]
        ] = []
        missing_phases: list[str] = []
        invalid_phases: list[str] = []
        video_excluded_phases: list[str] = []

        for (
            phase,
            start_column,
            end_column,
        ) in explicit_phase_specs:
            start_value = trial.get(
                start_column,
                pd.NA,
            )
            end_value = trial.get(
                end_column,
                pd.NA,
            )

            if (
                pd.isna(start_value)
                or pd.isna(end_value)
            ):
                missing_phases.append(phase)
                continue

            (
                start_utc_ms,
                start_rec_ms,
                start_video_ms,
            ) = convert_ns(start_value)
            (
                end_utc_ms,
                end_rec_ms,
                end_video_ms,
            ) = convert_ns(end_value)

            if (
                not np.isfinite(start_video_ms)
                or not np.isfinite(end_video_ms)
                or end_video_ms <= start_video_ms
            ):
                invalid_phases.append(phase)
                continue

            if (
                bool(cfg.require_full_phase_video_coverage)
                and np.isfinite(video_duration_ms)
            ):
                tolerance_ms = max(
                    float(cfg.video_coverage_tolerance_ms),
                    0.0,
                )
                if (
                    start_video_ms < -tolerance_ms
                    or end_video_ms
                    > float(video_duration_ms) + tolerance_ms
                ):
                    video_excluded_phases.append(phase)
                    continue

                # Only absorb tiny alignment differences within tolerance.
                start_video_ms = max(
                    0.0,
                    float(start_video_ms),
                )
                end_video_ms = min(
                    float(video_duration_ms),
                    float(end_video_ms),
                )
                if end_video_ms <= start_video_ms:
                    video_excluded_phases.append(phase)
                    continue

            explicit_rows.append(
                {
                    "trial_id": trial_id,
                    "trial_index": trial_index,
                    "task_id": task_id,
                    "task": f"Task{task_id}",
                    "phase": phase,
                    "phase_level": "fine"
                    if schema == "fine"
                    else "coarse",
                    "analysis_eligible": True,
                    "start_utc_ms": start_utc_ms,
                    "end_utc_ms": end_utc_ms,
                    "start_utc_datetime": (
                        pd.to_datetime(
                            start_utc_ms,
                            unit="ms",
                            utc=True,
                            errors="coerce",
                        )
                    ),
                    "end_utc_datetime": (
                        pd.to_datetime(
                            end_utc_ms,
                            unit="ms",
                            utc=True,
                            errors="coerce",
                        )
                    ),
                    "start_rec_ms": start_rec_ms,
                    "end_rec_ms": end_rec_ms,
                    "start_video_ms": start_video_ms,
                    "end_video_ms": end_video_ms,
                    "start_ms": start_video_ms,
                    "end_ms": end_video_ms,
                    "duration_ms": (
                        end_video_ms
                        - start_video_ms
                    ),
                    "source_start_column": (
                        start_column
                    ),
                    "source_end_column": end_column,
                    "timeline_clock_offset_ms": (
                        float(
                            cfg.timeline_clock_offset_ms
                        )
                    ),
                    "timeline_clock_drift_ppm": (
                        float(
                            cfg.timeline_clock_drift_ppm
                        )
                    ),
                    "recording_origin_utc_ms": (
                        float(origin_utc_ms)
                    ),
                    "video_start_rec_ms": (
                        float(video_start_rec_ms)
                    ),
                    "source_timeline_file": (
                        str(cfg.timeline_file or "")
                    ),
                    "timeline_schema": schema,
                    "user_id": int(
                        trial["user_id"]
                    ),
                    "block_press_count": (
                        block_press_count
                    ),
                    "warning_note": warning_note,
                    "user_csv_file": trial.get(
                        "user_csv_file",
                        "",
                    ),
                    "robot_log_file": trial.get(
                        "robot_log_file",
                        "",
                    ),
                    "match_note": trial.get(
                        "match_note",
                        "",
                    ),
                }
            )

        if not explicit_rows:
            qa_rows.append(
                {
                    "trial_id": trial_id,
                    "trial_index": trial_index,
                    "task_id": task_id,
                    "timeline_schema": schema,
                    "block_press_count": (
                        block_press_count
                    ),
                    "warning_note": warning_note,
                    "qa_status": "EXCLUDE",
                    "missing_phases": ",".join(
                        missing_phases
                    ),
                    "invalid_phases": ",".join(
                        invalid_phases
                    ),
                    "video_excluded_phases": ",".join(
                        video_excluded_phases
                    ),
                    "qa_issues": (
                        "no_usable_phase_intervals"
                        + (
                            ";missing="
                            + ",".join(missing_phases)
                            if missing_phases
                            else ""
                        )
                        + (
                            ";invalid="
                            + ",".join(invalid_phases)
                            if invalid_phases
                            else ""
                        )
                        + (
                            ";outside_video="
                            + ",".join(
                                video_excluded_phases
                            )
                            if video_excluded_phases
                            else ""
                        )
                    ),
                }
            )
            continue

        phase_by_name = {
            row["phase"]: row
            for row in explicit_rows
        }

        derived_rows: list[
            dict[str, Any]
        ] = []

        def make_parent_phase(
            phase: str,
            child_phase_names: list[str],
        ) -> dict[str, Any] | None:
            children = [
                phase_by_name[name]
                for name in child_phase_names
                if name in phase_by_name
            ]
            if not children:
                return None

            start_child = min(
                children,
                key=lambda item: (
                    item["start_video_ms"]
                ),
            )
            end_child = max(
                children,
                key=lambda item: (
                    item["end_video_ms"]
                ),
            )

            parent = dict(start_child)
            parent.update(
                {
                    "phase": phase,
                    "phase_level": "parent",
                    "analysis_eligible": True,
                    "start_utc_ms": (
                        start_child["start_utc_ms"]
                    ),
                    "end_utc_ms": (
                        end_child["end_utc_ms"]
                    ),
                    "start_utc_datetime": (
                        start_child[
                            "start_utc_datetime"
                        ]
                    ),
                    "end_utc_datetime": (
                        end_child[
                            "end_utc_datetime"
                        ]
                    ),
                    "start_rec_ms": (
                        start_child["start_rec_ms"]
                    ),
                    "end_rec_ms": (
                        end_child["end_rec_ms"]
                    ),
                    "start_video_ms": (
                        start_child["start_video_ms"]
                    ),
                    "end_video_ms": (
                        end_child["end_video_ms"]
                    ),
                    "start_ms": (
                        start_child["start_video_ms"]
                    ),
                    "end_ms": (
                        end_child["end_video_ms"]
                    ),
                    "duration_ms": (
                        end_child["end_video_ms"]
                        - start_child[
                            "start_video_ms"
                        ]
                    ),
                    "source_start_column": (
                        start_child[
                            "source_start_column"
                        ]
                    ),
                    "source_end_column": (
                        end_child[
                            "source_end_column"
                        ]
                    ),
                }
            )
            return parent

        if schema == "fine":
            t1_parent = make_parent_phase(
                "T1_Reference",
                [
                    "T1_Motion",
                    "T1_SafetyAssessment",
                    "T1_Placement",
                ],
            )
            t2_parent = make_parent_phase(
                "T2_Manipulated",
                [
                    "T2_Motion",
                    "T2_SafetyAssessment",
                    "T2_Placement",
                ],
            )
            if t1_parent is not None:
                derived_rows.append(t1_parent)
            if t2_parent is not None:
                derived_rows.append(t2_parent)

        all_analysis_rows = (
            explicit_rows
            + derived_rows
        )

        trial_start_row = min(
            all_analysis_rows,
            key=lambda item: (
                item["start_video_ms"]
            ),
        )
        trial_end_row = max(
            all_analysis_rows,
            key=lambda item: (
                item["end_video_ms"]
            ),
        )

        overall = dict(trial_start_row)
        overall.update(
            {
                "phase": "Trial_Overall",
                "phase_level": "overall",
                "analysis_eligible": True,
                "start_utc_ms": (
                    trial_start_row[
                        "start_utc_ms"
                    ]
                ),
                "end_utc_ms": (
                    trial_end_row[
                        "end_utc_ms"
                    ]
                ),
                "start_utc_datetime": (
                    trial_start_row[
                        "start_utc_datetime"
                    ]
                ),
                "end_utc_datetime": (
                    trial_end_row[
                        "end_utc_datetime"
                    ]
                ),
                "start_rec_ms": (
                    trial_start_row[
                        "start_rec_ms"
                    ]
                ),
                "end_rec_ms": (
                    trial_end_row[
                        "end_rec_ms"
                    ]
                ),
                "start_video_ms": (
                    trial_start_row[
                        "start_video_ms"
                    ]
                ),
                "end_video_ms": (
                    trial_end_row[
                        "end_video_ms"
                    ]
                ),
                "start_ms": (
                    trial_start_row[
                        "start_video_ms"
                    ]
                ),
                "end_ms": (
                    trial_end_row[
                        "end_video_ms"
                    ]
                ),
                "duration_ms": (
                    trial_end_row[
                        "end_video_ms"
                    ]
                    - trial_start_row[
                        "start_video_ms"
                    ]
                ),
                "source_start_column": (
                    trial_start_row[
                        "source_start_column"
                    ]
                ),
                "source_end_column": (
                    trial_end_row[
                        "source_end_column"
                    ]
                ),
            }
        )

        rows.append(overall)
        rows.extend(derived_rows)
        rows.extend(explicit_rows)

        issues: list[str] = []
        if missing_phases:
            issues.append(
                "missing="
                + ",".join(missing_phases)
            )
        if invalid_phases:
            issues.append(
                "invalid="
                + ",".join(invalid_phases)
            )
        if video_excluded_phases:
            issues.append(
                "outside_video="
                + ",".join(video_excluded_phases)
            )
        if warning_note:
            issues.append(
                "source_warning="
                + warning_note
            )

        phase_sequence = [
            phase
            for phase, _start, _end
            in explicit_phase_specs
        ]
        available_sequence_rows = [
            phase_by_name[phase]
            for phase in phase_sequence
            if phase in phase_by_name
        ]

        for first, second in zip(
            available_sequence_rows,
            available_sequence_rows[1:],
        ):
            if (
                first["end_video_ms"]
                > second["start_video_ms"]
                + 5.0
            ):
                issues.append(
                    "phase_overlap_or_order:"
                    + first["phase"]
                    + ">"
                    + second["phase"]
                )

        overall_start = float(
            overall["start_video_ms"]
        )
        overall_end = float(
            overall["end_video_ms"]
        )

        if np.isfinite(video_duration_ms):
            if overall_start < -1000.0:
                issues.append(
                    "trial_starts_before_video"
                )
            if (
                overall_end
                > video_duration_ms + 1000.0
            ):
                issues.append(
                    "trial_ends_after_video"
                )

        qa_status = (
            "OK"
            if not issues
            else (
                "PARTIAL"
                if explicit_rows
                else "EXCLUDE"
            )
        )

        qa_row = {
            "trial_id": trial_id,
            "trial_index": trial_index,
            "task_id": task_id,
            "timeline_schema": schema,
            "task_order_position": trial_index,
            "trial_start_video_ms": overall_start,
            "trial_end_video_ms": overall_end,
            "trial_duration_ms": (
                overall_end - overall_start
            ),
            "available_explicit_phase_count": (
                len(explicit_rows)
            ),
            "expected_explicit_phase_count": (
                len(explicit_phase_specs)
            ),
            "missing_phase_count": (
                len(missing_phases)
            ),
            "missing_phases": ",".join(
                missing_phases
            ),
            "invalid_phases": ",".join(
                invalid_phases
            ),
            "video_excluded_phases": ",".join(
                video_excluded_phases
            ),
            "video_coverage_required": bool(
                cfg.require_full_phase_video_coverage
            ),
            "video_coverage_tolerance_ms": float(
                cfg.video_coverage_tolerance_ms
            ),
            "block_press_count": (
                block_press_count
            ),
            "warning_note": warning_note,
            "video_duration_ms": (
                video_duration_ms
            ),
            "timeline_clock_offset_ms": (
                float(
                    cfg.timeline_clock_offset_ms
                )
            ),
            "timeline_clock_drift_ppm": (
                float(
                    cfg.timeline_clock_drift_ppm
                )
            ),
            "qa_status": qa_status,
            "qa_issues": ";".join(issues),
        }

        for phase_row in explicit_rows:
            token = (
                phase_row["phase"]
                .lower()
                .replace(
                    "safetyassessment",
                    "safety",
                )
            )
            qa_row[
                f"{token}_duration_ms"
            ] = phase_row["duration_ms"]

        qa_rows.append(qa_row)

    if not rows:
        raise ValueError(
            "No usable TOI rows could be generated "
            "for this participant."
        )

    phase_priority = {
        "Trial_Overall": 0,
        "T1_Reference": 1,
        "T1_Motion": 2,
        "T1_SafetyAssessment": 3,
        "T1_Placement": 4,
        "T2_Manipulated": 5,
        "T2_Motion": 6,
        "T2_SafetyAssessment": 7,
        "T2_Placement": 8,
        "T0_Baseline": 1,
    }

    toi = pd.DataFrame(rows)
    toi["_phase_priority"] = (
        toi["phase"]
        .map(phase_priority)
        .fillna(99)
    )
    toi = (
        toi.sort_values(
            [
                "trial_index",
                "_phase_priority",
                "start_video_ms",
            ]
        )
        .drop(
            columns=["_phase_priority"]
        )
        .reset_index(drop=True)
    )

    qa = pd.DataFrame(qa_rows)
    if not qa.empty:
        qa = qa.sort_values(
            "trial_index"
        ).reset_index(drop=True)

    toi_path = (
        out_dir
        / f"{cfg.output_prefix}_timeline_toi.csv"
    )
    qa_path = (
        out_dir
        / (
            f"{cfg.output_prefix}_"
            "timeline_alignment_QA.csv"
        )
    )
    write_csv(toi, toi_path)
    write_csv(qa, qa_path)

    print("\n  Timeline alignment:")
    print(
        f"    schema                  = "
        f"{schema}"
    )
    print(
        f"    recording origin UTC ms = "
        f"{origin_utc_ms:.3f}"
    )
    print(
        f"    video start rec ms      = "
        f"{video_start_rec_ms:.3f}"
    )
    print(
        f"    clock offset / drift    = "
        f"{cfg.timeline_clock_offset_ms:.3f} ms / "
        f"{cfg.timeline_clock_drift_ppm:.3f} ppm"
    )
    print(
        f"    generated TOI rows      = "
        f"{len(toi)}"
    )
    print(f"    TOI table               = {toi_path}")
    print(f"    QA table                = {qa_path}")

    if not qa.empty:
        review = qa[
            ~qa["qa_status"]
            .astype(str)
            .eq("OK")
        ]
        if not review.empty:
            print(
                "    WARNING: review required for: "
                + ", ".join(
                    review["trial_id"]
                    .astype(str)
                    .tolist()
                )
            )

    return toi, toi_path, qa_path


def export_raw_trial_clips(
    video_path: Path,
    toi: pd.DataFrame,
    out_dir: Path,
    cfg: PipelineConfig,
    video_props: dict[str, float],
) -> Path:
    """
    Export one raw MP4 clip for each Trial_Overall interval.

    ffmpeg is used when available; otherwise OpenCV performs exact-frame
    re-encoding without audio.
    """
    clips_dir = out_dir / "trial_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    overall = toi[
        toi["phase"].astype(str).eq("Trial_Overall")
    ].copy()
    if overall.empty:
        return clips_dir

    video_duration_ms = (
        float(video_props["frame_count"])
        / float(video_props["fps"])
        * 1000.0
    )
    ffmpeg = shutil.which("ffmpeg")

    for _, row in overall.iterrows():
        start_ms = max(
            0.0,
            float(row["start_video_ms"])
            - float(cfg.clip_padding_before_ms),
        )
        end_ms = min(
            video_duration_ms,
            float(row["end_video_ms"])
            + float(cfg.clip_padding_after_ms),
        )
        if end_ms <= start_ms:
            continue

        trial_index = int(row["trial_index"])
        task_id = int(row["task_id"])
        output_path = (
            clips_dir
            / f"{cfg.output_prefix}_Trial{trial_index:02d}_Task{task_id:02d}.mp4"
        )

        if ffmpeg:
            command = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_ms / 1000.0:.6f}",
                "-i",
                str(video_path),
                "-t",
                f"{(end_ms - start_ms) / 1000.0:.6f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                str(output_path),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                continue
            print(
                f"    ffmpeg failed for {output_path.name}; "
                "falling back to OpenCV."
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = float(video_props["fps"])
        width = int(video_props["width"])
        height = int(video_props["height"])
        start_frame = max(0, int(np.floor(start_ms / 1000.0 * fps)))
        end_frame = min(
            int(video_props["frame_count"]),
            int(np.ceil(end_ms / 1000.0 * fps)),
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        for _frame_index in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)

        writer.release()
        cap.release()

    print(f"  Trial clips saved: {clips_dir}")
    return clips_dir


def read_triggered_tasks_export(path: Path) -> pd.DataFrame:
    """
    Read D-Lab's tab-delimited Triggered Tasks export.

    Expected original columns are normally:
    Task | Start Time | End Time | Duration
    """
    import io

    path = Path(path)
    raw = path.read_bytes()
    parsed: list[tuple[int, str, str | None, pd.DataFrame]] = []

    for encoding in ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030", "latin1"]:
        try:
            decoded = raw.decode(encoding)
        except Exception:
            continue
        if not decoded.strip():
            continue

        for separator in ["\t", ",", ";", None]:
            try:
                if separator is None:
                    df = pd.read_csv(
                        io.StringIO(decoded),
                        sep=None,
                        engine="python",
                        dtype=str,
                        on_bad_lines="skip",
                    )
                else:
                    df = pd.read_csv(
                        io.StringIO(decoded),
                        sep=separator,
                        engine="python",
                        dtype=str,
                        on_bad_lines="skip",
                    )
            except Exception:
                continue

            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            normalized = {normalize_column_token(c): c for c in df.columns}
            has_task = any(k in normalized for k in ["task", "taskname"])
            has_start = any(k in normalized for k in ["starttime", "start", "startms", "startrecms"])
            has_end = any(k in normalized for k in ["endtime", "end", "endms", "endrecms"])
            if not (has_task and has_start and has_end):
                continue

            score = 1000 + len(df.columns) * 10 + len(df)
            parsed.append((score, encoding, separator, df))

    if not parsed:
        raise ValueError(
            f"Could not parse D-Lab Triggered Tasks file: {path}\n"
            "Expected columns such as Task, Start Time, End Time, Duration."
        )

    parsed.sort(key=lambda item: item[0], reverse=True)
    _, encoding, separator, df = parsed[0]
    separator_name = {"\t": "TAB", ",": "COMMA", ";": "SEMICOLON", None: "AUTO"}.get(
        separator, repr(separator)
    )
    print("  Triggered Tasks parser:")
    print(f"    path      = {path}")
    print(f"    encoding  = {encoding}")
    print(f"    delimiter = {separator_name}")
    print(f"    shape     = {df.shape[0]} rows x {df.shape[1]} columns")

    normalized = {normalize_column_token(c): c for c in df.columns}

    def get_col(keys: list[str], required: bool = True) -> str | None:
        for key in keys:
            if key in normalized:
                return normalized[key]
        if required:
            raise ValueError(f"Triggered Tasks column not found. Tried: {keys}")
        return None

    task_col = get_col(["task", "taskname"])
    start_col = get_col(["starttime", "start", "startms", "startrecms"])
    end_col = get_col(["endtime", "end", "endms", "endrecms"])
    duration_col = get_col(
        ["duration", "durationms", "taskduration", "taskdurationms"],
        required=False,
    )

    out = pd.DataFrame()
    out["source_task_name"] = df[task_col].astype(str).str.strip()
    out["start_rec_ms"] = df[start_col].apply(parse_time_to_ms)
    out["end_rec_ms"] = df[end_col].apply(parse_time_to_ms)
    if duration_col is not None:
        out["duration_reported_ms"] = df[duration_col].apply(parse_time_to_ms)
    else:
        out["duration_reported_ms"] = np.nan

    out = out[
        out["start_rec_ms"].notna()
        & out["end_rec_ms"].notna()
        & (out["end_rec_ms"] > out["start_rec_ms"])
    ].copy()
    out = out.sort_values(["start_rec_ms", "end_rec_ms"]).reset_index(drop=True)
    out["source_row_index"] = np.arange(len(out), dtype=int)
    return out


def infer_task_phases(task_df: pd.DataFrame) -> pd.DataFrame:
    """Infer T1/T2 phase names when D-Lab tasks are simply motion/safety/placement."""
    out = task_df.copy()
    phases: list[str] = []
    cycles: list[int | float] = []
    occurrences: dict[str, int] = {}
    current_cycle = 0

    for name in out["source_task_name"].astype(str):
        token = normalize_column_token(name)
        occurrences[token] = occurrences.get(token, 0) + 1

        explicit_t = re.search(r"t\s*([0-9]+)", name, flags=re.I)
        if explicit_t:
            cycle = int(explicit_t.group(1))
            current_cycle = max(current_cycle, cycle)
        else:
            cycle = current_cycle

        if "baseline" in token or token in {"t0", "base"}:
            phase = "T0_Baseline"
            cycle = 0
        elif "motion" in token:
            if not explicit_t:
                current_cycle += 1
                cycle = current_cycle
            phase = f"T{cycle}_Motion"
        elif "safetyassessment" in token or ("safety" in token and "assessment" in token):
            if cycle <= 0:
                cycle = max(1, current_cycle)
                current_cycle = cycle
            phase = f"T{cycle}_SafetyAssessment"
        elif "placement" in token or "place" in token:
            if cycle <= 0:
                cycle = max(1, current_cycle)
                current_cycle = cycle
            phase = f"T{cycle}_Placement"
        else:
            cleaned = re.sub(r"\s+", "_", name.strip())
            phase = cleaned if cleaned else f"Task_{len(phases) + 1}"
            cycle = np.nan

        phases.append(phase)
        cycles.append(cycle)

    out["task"] = out["source_task_name"]
    out["phase"] = phases
    out["task_cycle"] = cycles
    out["task_occurrence"] = [
        occurrences.get(normalize_column_token(name), 1)
        for name in out["source_task_name"].astype(str)
    ]
    return out


def resolve_time_alignment(
    rec_time_ms: pd.Series,
    utc_ms_raw: pd.Series,
    video_props: dict[str, float],
    cfg: PipelineConfig,
    triggered_tasks_raw: pd.DataFrame | None,
) -> dict[str, Any]:
    rec = pd.to_numeric(rec_time_ms, errors="coerce")
    utc = pd.to_numeric(utc_ms_raw, errors="coerce")
    rec_valid = rec.dropna()

    rec_min = float(rec_valid.min()) if not rec_valid.empty else np.nan
    rec_max = float(rec_valid.max()) if not rec_valid.empty else np.nan
    video_duration_ms = (
        float(video_props["frame_count"]) / float(video_props["fps"]) * 1000.0
        if float(video_props.get("fps", 0)) > 0
        else np.nan
    )

    first_task_start = np.nan
    last_task_end = np.nan
    if triggered_tasks_raw is not None and not triggered_tasks_raw.empty:
        first_task_start = float(triggered_tasks_raw["start_rec_ms"].min())
        last_task_end = float(triggered_tasks_raw["end_rec_ms"].max())

    if cfg.video_start_rec_ms is not None:
        video_start_rec_ms = float(cfg.video_start_rec_ms)
        method = "explicit_video_start_rec_ms"
    elif abs(float(cfg.time_offset_ms)) > 1e-9:
        video_start_rec_ms = -float(cfg.time_offset_ms)
        method = "legacy_time_offset_ms"
    else:
        video_start_rec_ms = 0.0
        method = "auto_recording_zero"

        if np.isfinite(rec_min) and np.isfinite(first_task_start) and abs(rec_min - first_task_start) <= 2000:
            # Prefer the exact D-Lab task boundary over the first sampled gaze row.
            video_start_rec_ms = first_task_start
            method = "auto_task_start_matches_gaze_start"
        elif np.isfinite(rec_min) and rec_min > 1000:
            video_start_rec_ms = rec_min
            method = "auto_first_gaze_rec_time"
        elif np.isfinite(first_task_start) and np.isfinite(last_task_end) and np.isfinite(video_duration_ms):
            task_window = last_task_end - first_task_start
            error_if_task_clip = abs(video_duration_ms - task_window)
            error_if_full_recording = abs(video_duration_ms - last_task_end)
            if error_if_task_clip + 1000 < error_if_full_recording:
                video_start_rec_ms = first_task_start
                method = "auto_video_duration_closer_to_task_window"

    pair_mask = rec.notna() & utc.notna() & (utc > 1e11)
    if pair_mask.any():
        offsets = utc[pair_mask] - rec[pair_mask]
        recording_origin_utc_ms = float(offsets.median())
        errors = offsets - recording_origin_utc_ms
        utc_pair_count = int(pair_mask.sum())
        utc_median_abs_error_ms = float(errors.abs().median())
        utc_max_abs_error_ms = float(errors.abs().max())
    else:
        recording_origin_utc_ms = np.nan
        utc_pair_count = 0
        utc_median_abs_error_ms = np.nan
        utc_max_abs_error_ms = np.nan

    return {
        "video_start_rec_ms": float(video_start_rec_ms),
        "video_start_alignment_method": method,
        "recording_origin_utc_ms": recording_origin_utc_ms,
        "utc_pair_count": utc_pair_count,
        "utc_median_abs_error_ms": utc_median_abs_error_ms,
        "utc_max_abs_error_ms": utc_max_abs_error_ms,
        "gaze_rec_time_min_ms": rec_min,
        "gaze_rec_time_max_ms": rec_max,
        "first_triggered_task_start_rec_ms": first_task_start,
        "last_triggered_task_end_rec_ms": last_task_end,
        "video_duration_ms": video_duration_ms,
    }


def convert_triggered_tasks_to_toi(
    triggered_tasks_path: Path,
    alignment: dict[str, Any],
    out_dir: Path,
    output_prefix: str,
) -> tuple[pd.DataFrame, Path]:
    raw = read_triggered_tasks_export(triggered_tasks_path)
    toi = infer_task_phases(raw)

    video_start_rec_ms = float(alignment["video_start_rec_ms"])
    origin_utc_ms = pd.to_numeric(
        pd.Series([alignment.get("recording_origin_utc_ms")]),
        errors="coerce",
    ).iloc[0]

    toi["start_video_ms"] = toi["start_rec_ms"] - video_start_rec_ms
    toi["end_video_ms"] = toi["end_rec_ms"] - video_start_rec_ms
    toi["duration_ms"] = toi["end_rec_ms"] - toi["start_rec_ms"]

    # Backward-compatible aliases used internally by compute_toi_metrics().
    toi["start_ms"] = toi["start_video_ms"]
    toi["end_ms"] = toi["end_video_ms"]

    if pd.notna(origin_utc_ms):
        toi["start_utc_ms"] = origin_utc_ms + toi["start_rec_ms"]
        toi["end_utc_ms"] = origin_utc_ms + toi["end_rec_ms"]
        toi["start_utc_datetime"] = pd.to_datetime(
            toi["start_utc_ms"], unit="ms", errors="coerce", utc=True
        ).astype(str)
        toi["end_utc_datetime"] = pd.to_datetime(
            toi["end_utc_ms"], unit="ms", errors="coerce", utc=True
        ).astype(str)
    else:
        toi["start_utc_ms"] = np.nan
        toi["end_utc_ms"] = np.nan
        toi["start_utc_datetime"] = ""
        toi["end_utc_datetime"] = ""

    toi["video_start_rec_ms"] = video_start_rec_ms
    toi["recording_origin_utc_ms"] = origin_utc_ms
    toi["source_triggered_tasks_file"] = str(triggered_tasks_path)

    ordered = [
        "task", "source_task_name", "phase", "task_cycle", "task_occurrence",
        "start_rec_ms", "end_rec_ms",
        "start_video_ms", "end_video_ms",
        "start_utc_ms", "end_utc_ms",
        "start_utc_datetime", "end_utc_datetime",
        "duration_ms", "duration_reported_ms",
        "start_ms", "end_ms",
        "video_start_rec_ms", "recording_origin_utc_ms",
        "source_row_index", "source_triggered_tasks_file",
    ]
    toi = toi[[column for column in ordered if column in toi.columns]].copy()

    out_path = out_dir / f"{output_prefix}_triggered_tasks_converted.csv"
    write_csv(toi, out_path)
    print(f"  converted Triggered Tasks saved: {out_path}")
    return toi, out_path


def annotate_samples_with_toi(samples: pd.DataFrame, toi: pd.DataFrame) -> pd.DataFrame:
    out = samples.copy()
    n = len(out)
    out["toi_row_index"] = pd.Series([pd.NA] * n, dtype="Int64")
    out["task"] = pd.Series([""] * n, dtype="object")
    out["phase"] = pd.Series([""] * n, dtype="object")
    out["toi_start_rec_ms"] = np.nan
    out["toi_end_rec_ms"] = np.nan
    out["toi_start_video_ms"] = np.nan
    out["toi_end_video_ms"] = np.nan
    out["toi_start_utc_ms"] = np.nan
    out["toi_end_utc_ms"] = np.nan

    time_col = "video_time_ms" if "video_time_ms" in out.columns else "gaze_time_ms"
    times = pd.to_numeric(out[time_col], errors="coerce")

    annotation_toi = toi.copy()
    annotation_toi["_annotation_duration_ms"] = (
        pd.to_numeric(
            annotation_toi.get(
                "end_video_ms",
                annotation_toi.get("end_ms"),
            ),
            errors="coerce",
        )
        - pd.to_numeric(
            annotation_toi.get(
                "start_video_ms",
                annotation_toi.get("start_ms"),
            ),
            errors="coerce",
        )
    )
    # Long parent windows are written first; shorter fine phases overwrite
    # them, so sample-level phase labels remain specific.
    annotation_toi = annotation_toi.sort_values(
        "_annotation_duration_ms",
        ascending=False,
    ).drop(
        columns=["_annotation_duration_ms"]
    )

    for idx, row in annotation_toi.reset_index(drop=True).iterrows():
        start = pd.to_numeric(pd.Series([row.get("start_video_ms", row.get("start_ms"))]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([row.get("end_video_ms", row.get("end_ms"))]), errors="coerce").iloc[0]
        if pd.isna(start) or pd.isna(end):
            continue
        mask = times.ge(float(start)) & times.lt(float(end))
        out.loc[mask, "toi_row_index"] = int(idx)
        out.loc[mask, "task"] = str(row.get("task", row.get("source_task_name", "")))
        out.loc[mask, "phase"] = str(row.get("phase", ""))
        out.loc[mask, "toi_start_rec_ms"] = row.get("start_rec_ms", np.nan)
        out.loc[mask, "toi_end_rec_ms"] = row.get("end_rec_ms", np.nan)
        out.loc[mask, "toi_start_video_ms"] = start
        out.loc[mask, "toi_end_video_ms"] = end
        out.loc[mask, "toi_start_utc_ms"] = row.get("start_utc_ms", np.nan)
        out.loc[mask, "toi_end_utc_ms"] = row.get("end_utc_ms", np.nan)

    return out


def classify_gaze(
    gaze_path: Path,
    aoi_df: pd.DataFrame,
    out_dir: Path,
    cfg: PipelineConfig,
    video_props: dict[str, float],
    processing_windows: pd.DataFrame | None = None,
    precomputed_alignment: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path, dict[str, Any]]:
    print("\n[3/5] Classify gaze into dynamic AOIs")
    gaze_df = read_table_auto(gaze_path)
    time_col = choose_column(gaze_df, cfg.time_col, DEFAULT_TIME_CANDIDATES, "time")
    gaze_x_col = choose_column(gaze_df, cfg.gaze_x_col, DEFAULT_GAZE_X_CANDIDATES, "gaze_x")
    gaze_y_col = choose_column(gaze_df, cfg.gaze_y_col, DEFAULT_GAZE_Y_CANDIDATES, "gaze_y")
    utc_col = next((c for c in ["UTC", "utc", "Utc"] if c in gaze_df.columns), None)

    rec_time_series = gaze_df[time_col].apply(parse_time_to_ms)
    utc_numeric_series = (
        pd.to_numeric(gaze_df[utc_col], errors="coerce")
        if utc_col is not None
        else pd.Series(np.nan, index=gaze_df.index)
    )
    triggered_tasks_raw = (
        read_triggered_tasks_export(cfg.triggered_tasks_txt)
        if cfg.triggered_tasks_txt is not None
        else None
    )
    alignment = (
        dict(precomputed_alignment)
        if precomputed_alignment is not None
        else resolve_time_alignment(
            rec_time_ms=rec_time_series,
            utc_ms_raw=utc_numeric_series,
            video_props=video_props,
            cfg=cfg,
            triggered_tasks_raw=triggered_tasks_raw,
        )
    )

    original_source_rows = pd.Series(
        np.arange(len(gaze_df)),
        index=gaze_df.index,
    )

    if processing_windows is not None and not processing_windows.empty:
        video_time_all = (
            rec_time_series
            - float(alignment["video_start_rec_ms"])
        )
        keep_mask = pd.Series(
            False,
            index=gaze_df.index,
        )
        for _, window in processing_windows.iterrows():
            keep_mask = keep_mask | (
                video_time_all.ge(
                    float(window["processing_start_video_ms"])
                )
                & video_time_all.le(
                    float(window["processing_end_video_ms"])
                )
            )

        original_count = len(gaze_df)
        gaze_df = gaze_df.loc[keep_mask].copy()
        rec_time_series = rec_time_series.loc[keep_mask].copy()
        utc_numeric_series = utc_numeric_series.loc[keep_mask].copy()
        original_source_rows = (
            original_source_rows.loc[keep_mask].copy()
        )

        gaze_df = gaze_df.reset_index(drop=True)
        rec_time_series = rec_time_series.reset_index(drop=True)
        utc_numeric_series = utc_numeric_series.reset_index(drop=True)
        original_source_rows = original_source_rows.reset_index(
            drop=True
        )

        print(
            f"  gaze rows retained in trial windows: "
            f"{len(gaze_df)} / {original_count} "
            f"({len(gaze_df) / original_count:.1%})"
        )

    aoi_frame_width = float(cfg.aoi_frame_width or video_props["width"])
    aoi_frame_height = float(cfg.aoi_frame_height or video_props["height"])
    scale_x = 1.0 if cfg.disable_scaling else aoi_frame_width / float(cfg.gaze_native_width)
    scale_y = 1.0 if cfg.disable_scaling else aoi_frame_height / float(cfg.gaze_native_height)

    print(f"  time_col   = {time_col}")
    print(f"  gaze_x_col = {gaze_x_col}")
    print(f"  gaze_y_col = {gaze_y_col}")
    print(f"  gaze native frame = {cfg.gaze_native_width} x {cfg.gaze_native_height}")
    print(f"  AOI/video frame   = {aoi_frame_width} x {aoi_frame_height}")
    print(f"  scaling enabled   = {not cfg.disable_scaling}")
    print(f"  scale_x, scale_y  = {scale_x:.6f}, {scale_y:.6f}")

    samples = pd.DataFrame()
    samples["source_row"] = original_source_rows.astype(int)
    samples["raw_time"] = gaze_df[time_col]

    # Keep the original D-Lab recording clock intact.
    samples["rec_time_ms"] = rec_time_series
    samples["video_start_rec_ms"] = float(alignment["video_start_rec_ms"])
    samples["video_time_ms"] = (
        samples["rec_time_ms"] - samples["video_start_rec_ms"]
    )

    # Backward-compatible names: all AOI/video operations use video time.
    samples["gaze_time_ms_raw"] = samples["rec_time_ms"]
    samples["gaze_time_ms"] = samples["video_time_ms"]

    samples["gaze_x_raw_text"] = gaze_df[gaze_x_col]
    samples["gaze_y_raw_text"] = gaze_df[gaze_y_col]
    samples["gaze_x_raw"] = clean_numeric(gaze_df[gaze_x_col])
    samples["gaze_y_raw"] = clean_numeric(gaze_df[gaze_y_col])
    samples["gaze_x"] = samples["gaze_x_raw"] * scale_x
    samples["gaze_y"] = samples["gaze_y_raw"] * scale_y

    pupil_cols = detect_pupil_columns(gaze_df)
    samples["pupil_area_mean"] = row_mean_numeric(gaze_df, pupil_cols["pupil_area"])
    samples["pupil_height_mean"] = row_mean_numeric(gaze_df, pupil_cols["pupil_height"])
    samples["pupil_width_mean"] = row_mean_numeric(gaze_df, pupil_cols["pupil_width"])

    pupil_area_positive = pd.to_numeric(
        samples["pupil_area_mean"], errors="coerce"
    ).where(lambda value: value > 0)
    samples["pupil_equivalent_diameter_px"] = (
        2.0 * np.sqrt(pupil_area_positive / np.pi)
    )

    eye_pupil_cols = detect_eye_pupil_source_columns(gaze_df)
    samples["left_pupil_available_row"] = any_positive_numeric(
        gaze_df, eye_pupil_cols["left"]
    )
    samples["right_pupil_available_row"] = any_positive_numeric(
        gaze_df, eye_pupil_cols["right"]
    )
    samples["both_pupils_available_row"] = (
        samples["left_pupil_available_row"]
        & samples["right_pupil_available_row"]
    )

    if utc_col is not None:
        samples["UTC"] = gaze_df[utc_col]
        samples["utc_ms_raw"] = utc_numeric_series
    else:
        samples["UTC"] = ""
        samples["utc_ms_raw"] = np.nan

    origin_utc_ms = alignment.get("recording_origin_utc_ms", np.nan)
    samples["recording_origin_utc_ms"] = origin_utc_ms
    if pd.notna(origin_utc_ms):
        samples["utc_ms_reconstructed"] = origin_utc_ms + samples["rec_time_ms"]
    else:
        samples["utc_ms_reconstructed"] = np.nan

    samples["utc_ms"] = samples["utc_ms_raw"].combine_first(
        samples["utc_ms_reconstructed"]
    )
    samples["utc_alignment_error_ms"] = (
        samples["utc_ms_raw"] - samples["utc_ms_reconstructed"]
    )
    samples["utc_datetime"] = pd.to_datetime(
        samples["utc_ms"], unit="ms", errors="coerce", utc=True
    )

    samples["time_valid"] = samples["gaze_time_ms"].notna()
    samples["coord_numeric"] = samples["gaze_x_raw"].notna() & samples["gaze_y_raw"].notna()
    samples["coord_in_gaze_native_frame"] = (
        samples["coord_numeric"]
        & (samples["gaze_x_raw"] > 0)
        & (samples["gaze_y_raw"] > 0)
        & (samples["gaze_x_raw"] <= cfg.gaze_native_width)
        & (samples["gaze_y_raw"] <= cfg.gaze_native_height)
    )
    samples["coord_in_aoi_frame_after_scaling"] = (
        samples["coord_numeric"]
        & (samples["gaze_x"] > 0)
        & (samples["gaze_y"] > 0)
        & (samples["gaze_x"] <= aoi_frame_width)
        & (samples["gaze_y"] <= aoi_frame_height)
    )
    samples["gaze_quality_status"] = assign_gaze_quality_status(
        samples=samples,
        gaze_x_raw_text=samples["gaze_x_raw_text"],
        gaze_y_raw_text=samples["gaze_y_raw_text"],
        gaze_native_width=float(cfg.gaze_native_width),
        gaze_native_height=float(cfg.gaze_native_height),
        aoi_frame_width=float(aoi_frame_width),
        aoi_frame_height=float(aoi_frame_height),
    )
    samples["valid_gaze"] = samples["gaze_quality_status"].eq("ValidGaze")

    # Keep raw export-row durations for diagnostics, but use a canonical
    # valid-gaze stream for all AOI and pupil duration metrics.
    samples["raw_row_dt_ms"] = compute_dt_ms(samples["gaze_time_ms"])
    samples["analysis_dt_ms"] = assign_valid_stream_durations(
        samples,
        continuity_gap_ms=float(cfg.gaze_continuity_gap_ms),
    )
    samples["dt_ms"] = samples["analysis_dt_ms"]

    aoi_timestamps, aoi_lookup = build_aoi_lookup(aoi_df)
    matched_ts: list[float] = []
    matched_diff: list[float] = []
    aoi_available_flags: list[bool] = []
    labels: list[str] = []
    strict_labels: list[str] = []
    robot_near_flags: list[bool] = []
    robot_near_sources: list[str] = []
    robot_near_distances: list[float] = []

    for _, row in samples.iterrows():
        if not bool(row["valid_gaze"]):
            matched_ts.append(np.nan)
            matched_diff.append(np.nan)
            aoi_available_flags.append(False)
            labels.append("InvalidGaze")
            strict_labels.append("InvalidGaze")
            robot_near_flags.append(False)
            robot_near_sources.append("")
            robot_near_distances.append(np.nan)
            continue
        ts, diff = nearest_timestamp(row["gaze_time_ms"], aoi_timestamps, cfg.max_nearest_ms)
        matched_ts.append(ts if ts is not None else np.nan)
        matched_diff.append(diff)
        if ts is None:
            aoi_available_flags.append(False)
            labels.append("NoAOIData")
            strict_labels.append("NoAOIData")
            robot_near_flags.append(False)
            robot_near_sources.append("")
            robot_near_distances.append(np.nan)
            continue

        (
            label,
            strict_label,
            aoi_avail,
            robot_near_flag,
            robot_near_source,
            robot_near_distance,
        ) = classify_one_gaze_detailed(
            float(row["gaze_x"]),
            float(row["gaze_y"]),
            aoi_lookup[int(ts)],
            robot_near_enabled=bool(
                cfg.robot_near_enabled
            ),
            gripper_margin_px=float(
                cfg.robot_near_gripper_margin_px
            ),
            robot_arm_margin_px=float(
                cfg.robot_near_arm_margin_px
            ),
        )
        aoi_available_flags.append(aoi_avail)
        labels.append(label)
        strict_labels.append(strict_label)
        robot_near_flags.append(robot_near_flag)
        robot_near_sources.append(
            robot_near_source
        )
        robot_near_distances.append(
            robot_near_distance
        )

    samples["matched_aoi_timestamp_ms"] = matched_ts
    samples["aoi_time_diff_ms"] = matched_diff
    samples["aoi_available"] = aoi_available_flags
    samples["aoi_label_strict"] = strict_labels
    samples["robot_near"] = robot_near_flags
    samples["robot_near_source"] = robot_near_sources
    samples["robot_near_distance_px"] = robot_near_distances
    samples["aoi_label"] = labels

    row_quality_summary = build_row_quality_summary(samples)
    (
        gaze_stream_quality_df,
        gaze_loss_runs_df,
        gaze_stream_quality,
    ) = estimate_gaze_stream_quality(
        samples,
        continuity_gap_ms=float(cfg.gaze_continuity_gap_ms),
    )

    gaze_stream_quality["left_pupil_available_export_row_ratio"] = (
        float(samples["left_pupil_available_row"].mean()) if len(samples) else np.nan
    )
    gaze_stream_quality["right_pupil_available_export_row_ratio"] = (
        float(samples["right_pupil_available_row"].mean()) if len(samples) else np.nan
    )
    gaze_stream_quality["both_pupils_available_export_row_ratio"] = (
        float(samples["both_pupils_available_row"].mean()) if len(samples) else np.nan
    )
    gaze_stream_quality_df = pd.DataFrame([gaze_stream_quality])

    metrics_df, label_summary = compute_metrics(samples)

    out_samples = out_dir / f"{cfg.output_prefix}_samples_with_aoi.csv"
    out_metrics = out_dir / f"{cfg.output_prefix}_aoi_metrics_overall.csv"
    out_label_summary = out_dir / f"{cfg.output_prefix}_aoi_label_summary.csv"
    out_row_quality = out_dir / f"{cfg.output_prefix}_gaze_export_row_quality.csv"
    out_stream_quality = out_dir / f"{cfg.output_prefix}_gaze_stream_quality.csv"
    out_loss_runs = out_dir / f"{cfg.output_prefix}_gaze_tracking_loss_runs.csv"
    out_alignment = out_dir / f"{cfg.output_prefix}_time_alignment.csv"

    write_csv(samples, out_samples)
    write_csv(metrics_df, out_metrics)
    write_csv(label_summary, out_label_summary)
    write_csv(row_quality_summary, out_row_quality)
    write_csv(gaze_stream_quality_df, out_stream_quality)
    write_csv(gaze_loss_runs_df, out_loss_runs)
    write_csv(pd.DataFrame([alignment]), out_alignment)

    diagnostics = {
        "gaze_rows": len(samples),
        "parsed_time_rows": int(samples["time_valid"].sum()),
        "video_start_rec_ms": alignment.get("video_start_rec_ms", np.nan),
        "video_start_alignment_method": alignment.get("video_start_alignment_method", ""),
        "recording_origin_utc_ms": alignment.get("recording_origin_utc_ms", np.nan),
        "utc_pair_count": alignment.get("utc_pair_count", 0),
        "utc_median_abs_error_ms": alignment.get("utc_median_abs_error_ms", np.nan),
        "utc_max_abs_error_ms": alignment.get("utc_max_abs_error_ms", np.nan),
        "rec_time_min_ms": float(samples["rec_time_ms"].min()) if samples["rec_time_ms"].notna().any() else np.nan,
        "rec_time_max_ms": float(samples["rec_time_ms"].max()) if samples["rec_time_ms"].notna().any() else np.nan,
        "video_time_min_ms": float(samples["video_time_ms"].min()) if samples["video_time_ms"].notna().any() else np.nan,
        "video_time_max_ms": float(samples["video_time_ms"].max()) if samples["video_time_ms"].notna().any() else np.nan,
        "numeric_gaze_rows": int(samples["coord_numeric"].sum()),
        "in_native_frame_gaze_rows": int(samples["coord_in_gaze_native_frame"].sum()),
        "in_aoi_frame_after_scaling_rows": int(samples["coord_in_aoi_frame_after_scaling"].sum()),
        "valid_gaze_rows": int(samples["valid_gaze"].sum()),
        "gaze_time_min_ms": float(samples["gaze_time_ms"].min()) if samples["gaze_time_ms"].notna().any() else np.nan,
        "gaze_time_max_ms": float(samples["gaze_time_ms"].max()) if samples["gaze_time_ms"].notna().any() else np.nan,
        "aoi_time_min_ms": int(aoi_timestamps.min()) if len(aoi_timestamps) else np.nan,
        "aoi_time_max_ms": int(aoi_timestamps.max()) if len(aoi_timestamps) else np.nan,
        "pupil_area_columns": ";".join(pupil_cols["pupil_area"]),
        "pupil_height_columns": ";".join(pupil_cols["pupil_height"]),
        "pupil_width_columns": ";".join(pupil_cols["pupil_width"]),
        "left_pupil_source_columns": ";".join(eye_pupil_cols["left"]),
        "right_pupil_source_columns": ";".join(eye_pupil_cols["right"]),
        "estimated_valid_gaze_coverage_ratio": gaze_stream_quality.get(
            "estimated_valid_gaze_coverage_ratio", np.nan
        ),
        "estimated_tracking_loss_ratio": gaze_stream_quality.get(
            "estimated_tracking_loss_ratio", np.nan
        ),
        "longest_estimated_tracking_loss_run_ms": gaze_stream_quality.get(
            "longest_estimated_tracking_loss_run_ms", np.nan
        ),
        "tracking_loss_run_count_over_500ms": gaze_stream_quality.get(
            "tracking_loss_run_count_over_500ms", 0
        ),
        "tracking_loss_run_count_over_1000ms": gaze_stream_quality.get(
            "tracking_loss_run_count_over_1000ms", 0
        ),
        "gaze_continuity_gap_threshold_ms": float(cfg.gaze_continuity_gap_ms),
        "robot_near_enabled": bool(cfg.robot_near_enabled),
        "robot_near_gripper_margin_px": float(
            cfg.robot_near_gripper_margin_px
        ),
        "robot_near_arm_margin_px": float(
            cfg.robot_near_arm_margin_px
        ),
        "robot_near_valid_row_count": int(
            samples["robot_near"].sum()
        ),
    }
    diagnostics_csv = out_dir / f"{cfg.output_prefix}_diagnostics.csv"
    write_csv(pd.DataFrame([diagnostics]), diagnostics_csv)

    print("\n  Diagnostics:")
    for k, v in diagnostics.items():
        if not str(k).endswith("columns"):
            print(f"    {k}: {v}")
    print("\n  AOI label summary:")
    print(label_summary.to_string(index=False))
    print("\n  Pupil summary:")
    pupil_print_columns = [
        column
        for column in [
            "pupil_area_mean",
            "pupil_height_mean",
            "pupil_width_mean",
            "pupil_equivalent_diameter_mean_px",
            "pupil_equivalent_diameter_peak95_px",
            "pupil_equivalent_diameter_slope_px_per_s",
        ]
        if column in metrics_df.columns
    ]
    if pupil_print_columns:
        print(metrics_df[pupil_print_columns].to_string(index=False))
    print("\n  Gaze-stream quality estimate:")
    print(gaze_stream_quality_df.to_string(index=False))
    print("\n  Important:")
    print("    The old InvalidGaze row category is retained for compatibility.")
    print("    Use estimated_tracking_loss_ratio from *_gaze_stream_quality.csv")
    print("    when assessing actual gaze-stream loss.")
    print(f"  saved: {out_samples}")
    print(f"  saved: {out_metrics}")
    print(f"  saved: {out_label_summary}")
    print(f"  saved: {out_row_quality}")
    print(f"  saved: {out_stream_quality}")
    print(f"  saved: {out_loss_runs}")
    print(f"  saved: {out_alignment}")
    return samples, metrics_df, label_summary, out_samples, out_metrics, out_label_summary, diagnostics


def collapse_consecutive(labels: list[str]) -> list[str]:
    out: list[str] = []
    prev = None
    for lab in labels:
        if lab != prev:
            out.append(lab)
            prev = lab
    return out


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str = "dt_ms") -> float:
    if value_col not in df.columns:
        return np.nan
    val = pd.to_numeric(df[value_col], errors="coerce")
    wt = pd.to_numeric(df[weight_col], errors="coerce")
    mask = val.notna() & wt.notna() & (wt > 0)
    if not mask.any():
        return np.nan
    return float((val[mask] * wt[mask]).sum() / wt[mask].sum())



def weighted_quantile(
    df: pd.DataFrame,
    column: str,
    quantile: float,
) -> float:
    if df.empty or column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce")
    weights = pd.to_numeric(df.get("dt_ms", 1.0), errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    values_np = values.loc[mask].to_numpy(dtype=float)
    weights_np = weights.loc[mask].to_numpy(dtype=float)
    order = np.argsort(values_np)
    values_np = values_np[order]
    weights_np = weights_np[order]
    cumulative = np.cumsum(weights_np)
    target = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, len(values_np) - 1)
    return float(values_np[index])


def weighted_linear_slope_per_second(
    df: pd.DataFrame,
    value_column: str,
) -> float:
    if df.empty or value_column not in df.columns:
        return np.nan
    time = pd.to_numeric(df.get("gaze_time_ms"), errors="coerce")
    value = pd.to_numeric(df[value_column], errors="coerce")
    weight = pd.to_numeric(df.get("dt_ms", 1.0), errors="coerce")
    mask = (
        time.notna()
        & value.notna()
        & weight.notna()
        & (weight > 0)
    )
    if int(mask.sum()) < 3:
        return np.nan
    x = (time.loc[mask].to_numpy(dtype=float) - float(time.loc[mask].min())) / 1000.0
    y = value.loc[mask].to_numpy(dtype=float)
    w = weight.loc[mask].to_numpy(dtype=float)
    if np.ptp(x) <= 0:
        return np.nan
    try:
        slope = np.polyfit(x, y, 1, w=np.sqrt(w))[0]
    except Exception:
        return np.nan
    return float(slope)


def compute_metrics(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_df = samples[
        samples["valid_gaze"]
        & (pd.to_numeric(samples["dt_ms"], errors="coerce") > 0)
    ].copy()
    valid_gaze_duration_ms = (
        float(valid_df["dt_ms"].sum()) if len(valid_df) else 0.0
    )

    interval_duration_attr = samples.attrs.get("interval_duration_ms", np.nan)
    if pd.notna(interval_duration_attr):
        total_duration_ms = float(interval_duration_attr)
    else:
        parsed_times = pd.to_numeric(
            samples.loc[samples["time_valid"], "gaze_time_ms"],
            errors="coerce",
        ).dropna()
        if parsed_times.empty:
            total_duration_ms = valid_gaze_duration_ms
        else:
            expected = (
                float(
                    pd.to_numeric(
                        valid_df["dt_ms"], errors="coerce"
                    ).replace(0, np.nan).median()
                )
                if not valid_df.empty
                else 16.7
            )
            if not np.isfinite(expected):
                expected = 16.7
            total_duration_ms = max(
                expected,
                float(parsed_times.max() - parsed_times.min() + expected),
            )

    estimated_tracking_loss_ms = max(
        0.0,
        total_duration_ms - valid_gaze_duration_ms,
    )
    classified_df = valid_df[
        valid_df["aoi_label"].isin(
            MAIN_AOIS + [ROBOT_NEAR_LABEL, "Outside"]
        )
    ].copy()
    classified_duration_ms = float(classified_df["dt_ms"].sum()) if len(classified_df) else 0.0

    metrics: dict[str, Any] = {
        "total_rows": int(len(samples)),
        "total_duration_ms": total_duration_ms,
        "valid_gaze_rows": int(samples["valid_gaze"].sum()) if "valid_gaze" in samples else 0,
        "valid_gaze_duration_ms": valid_gaze_duration_ms,
        "valid_gaze_ratio_by_duration": valid_gaze_duration_ms / total_duration_ms if total_duration_ms else np.nan,
        "classified_duration_ms": classified_duration_ms,
        "classified_ratio_of_valid_duration": classified_duration_ms / valid_gaze_duration_ms if valid_gaze_duration_ms else np.nan,
        "pupil_area_mean": weighted_mean(valid_df, "pupil_area_mean"),
        "pupil_height_mean": weighted_mean(valid_df, "pupil_height_mean"),
        "pupil_width_mean": weighted_mean(valid_df, "pupil_width_mean"),
        "pupil_equivalent_diameter_mean_px": weighted_mean(
            valid_df,
            "pupil_equivalent_diameter_px",
        ),
        "pupil_equivalent_diameter_peak95_px": weighted_quantile(
            valid_df,
            "pupil_equivalent_diameter_px",
            0.95,
        ),
        "pupil_equivalent_diameter_slope_px_per_s": (
            weighted_linear_slope_per_second(
                valid_df,
                "pupil_equivalent_diameter_px",
            )
        ),
        "estimated_tracking_loss_duration_ms": estimated_tracking_loss_ms,
        "estimated_tracking_loss_ratio": (
            estimated_tracking_loss_ms / total_duration_ms
            if total_duration_ms
            else np.nan
        ),
    }

    for aoi in MAIN_AOIS + [ROBOT_NEAR_LABEL, "Outside", "NoAOIData", "InvalidGaze"]:
        if aoi == "InvalidGaze":
            dur = estimated_tracking_loss_ms
        else:
            dur = (
                float(
                    samples.loc[
                        samples["aoi_label"].eq(aoi),
                        "dt_ms",
                    ].sum()
                )
                if len(samples)
                else 0.0
            )
        metrics[f"{aoi}_duration_ms"] = dur
        metrics[f"{aoi}_ratio_of_total_duration"] = dur / total_duration_ms if total_duration_ms else np.nan
        metrics[f"{aoi}_ratio_of_valid_gaze"] = (dur / valid_gaze_duration_ms if valid_gaze_duration_ms and aoi != "InvalidGaze" else np.nan)


    # RobotTotal strict: direct RobotArm + direct Gripper only.
    strict_label_column = (
        "aoi_label_strict"
        if "aoi_label_strict" in samples.columns
        else "aoi_label"
    )
    robot_total_strict_duration_ms = (
        float(
            samples.loc[
                samples[strict_label_column].isin(
                    ["RobotArm", "Gripper"]
                ),
                "dt_ms",
            ].sum()
        )
        if len(samples)
        else 0.0
    )

    # RobotTotal buffered: direct robot AOIs plus RobotNear.
    robot_total_buffered_duration_ms = (
        float(
            samples.loc[
                samples["aoi_label"].isin(
                    [
                        "RobotArm",
                        "Gripper",
                        ROBOT_NEAR_LABEL,
                    ]
                ),
                "dt_ms",
            ].sum()
        )
        if len(samples)
        else 0.0
    )

    metrics[
        "RobotTotal_Strict_duration_ms"
    ] = robot_total_strict_duration_ms
    metrics[
        "RobotTotal_Strict_ratio_of_total_duration"
    ] = (
        robot_total_strict_duration_ms
        / total_duration_ms
        if total_duration_ms
        else np.nan
    )
    metrics[
        "RobotTotal_Strict_ratio_of_valid_gaze"
    ] = (
        robot_total_strict_duration_ms
        / valid_gaze_duration_ms
        if valid_gaze_duration_ms
        else np.nan
    )

    metrics[
        "RobotTotal_Buffered_duration_ms"
    ] = robot_total_buffered_duration_ms
    metrics[
        "RobotTotal_Buffered_ratio_of_total_duration"
    ] = (
        robot_total_buffered_duration_ms
        / total_duration_ms
        if total_duration_ms
        else np.nan
    )
    metrics[
        "RobotTotal_Buffered_ratio_of_valid_gaze"
    ] = (
        robot_total_buffered_duration_ms
        / valid_gaze_duration_ms
        if valid_gaze_duration_ms
        else np.nan
    )

    # Backward-compatible RobotTotal columns use the buffered definition.
    metrics[
        "RobotTotal_duration_ms"
    ] = robot_total_buffered_duration_ms
    metrics[
        "RobotTotal_ratio_of_total_duration"
    ] = (
        robot_total_buffered_duration_ms
        / total_duration_ms
        if total_duration_ms
        else np.nan
    )
    metrics[
        "RobotTotal_ratio_of_valid_gaze"
    ] = (
        robot_total_buffered_duration_ms
        / valid_gaze_duration_ms
        if valid_gaze_duration_ms
        else np.nan
    )


    labels_main = samples.loc[samples["aoi_label"].isin(MAIN_AOIS), "aoi_label"].tolist() if len(samples) else []
    collapsed_main = collapse_consecutive(labels_main)
    switch_count_main = max(0, len(collapsed_main) - 1)
    metrics["switch_count_main_aoi"] = switch_count_main
    metrics["switch_frequency_per_s_valid"] = switch_count_main / (valid_gaze_duration_ms / 1000) if valid_gaze_duration_ms else np.nan

    rt_labels: list[str] = []
    for lab in samples["aoi_label"].tolist() if len(samples) else []:
        if lab in ["RobotArm", "Gripper", ROBOT_NEAR_LABEL]:
            rt_labels.append("Robot")
        elif lab in ["Task", "Delivery"]:
            rt_labels.append("TaskDelivery")
    collapsed_rt = collapse_consecutive(rt_labels)
    robot_task_switches = sum(1 for a, b in zip(collapsed_rt[:-1], collapsed_rt[1:]) if {a, b} == {"Robot", "TaskDelivery"})
    metrics["robot_to_taskdelivery_switch_count"] = robot_task_switches
    metrics["robot_to_taskdelivery_switch_frequency_per_s_valid"] = robot_task_switches / (valid_gaze_duration_ms / 1000) if valid_gaze_duration_ms else np.nan

    labels_all = collapse_consecutive(samples["aoi_label"].tolist() if len(samples) else [])
    for aoi in MAIN_AOIS:
        entry_count = sum(1 for lab in labels_all if lab == aoi)
        metrics[f"{aoi}_entry_count"] = entry_count
        metrics[f"{aoi}_revisit_count"] = max(0, entry_count - 1)

    robot_total_sequence = [
        "RobotTotal" if label in {"RobotArm", "Gripper", ROBOT_NEAR_LABEL} else label
        for label in samples["aoi_label"].tolist()
    ] if len(samples) else []
    robot_total_collapsed = collapse_consecutive(robot_total_sequence)
    robot_total_entry_count = sum(
        1 for label in robot_total_collapsed if label == "RobotTotal"
    )
    metrics["RobotTotal_entry_count"] = robot_total_entry_count
    metrics["RobotTotal_revisit_count"] = max(0, robot_total_entry_count - 1)

    # Dominant label among valid classified samples.
    label_durations = samples.groupby("aoi_label", dropna=False)["dt_ms"].sum() if len(samples) else pd.Series(dtype=float)
    valid_labels = [
        a
        for a in MAIN_AOIS + [ROBOT_NEAR_LABEL, "Outside"]
        if a in label_durations.index
    ]
    if valid_labels:
        dominant = max(valid_labels, key=lambda x: float(label_durations.get(x, 0)))
        metrics["dominant_aoi_label"] = dominant
        metrics["dominant_aoi_duration_ms"] = float(label_durations.get(dominant, 0))
    else:
        metrics["dominant_aoi_label"] = ""
        metrics["dominant_aoi_duration_ms"] = np.nan

    metrics_df = pd.DataFrame([metrics])

    if len(samples):
        positive_duration = samples[
            pd.to_numeric(samples["dt_ms"], errors="coerce") > 0
        ].copy()
        counts = (
            positive_duration["aoi_label"]
            .value_counts(dropna=False)
            .reset_index()
        )
        counts.columns = ["aoi_label", "row_count"]
        dur = (
            positive_duration.groupby("aoi_label", dropna=False)["dt_ms"]
            .sum()
            .reset_index(name="duration_ms")
        )
        label_summary = counts.merge(dur, on="aoi_label", how="outer")
    else:
        label_summary = pd.DataFrame(
            columns=["aoi_label", "row_count", "duration_ms"]
        )

    # Replace raw blank-row InvalidGaze with estimated temporal tracking loss.
    label_summary = label_summary[
        ~label_summary["aoi_label"].astype(str).eq("InvalidGaze")
    ].copy()
    invalid_export_row_count = (
        int((~samples["valid_gaze"]).sum()) if len(samples) else 0
    )
    label_summary = pd.concat(
        [
            label_summary,
            pd.DataFrame(
                [
                    {
                        "aoi_label": "InvalidGaze",
                        "row_count": invalid_export_row_count,
                        "duration_ms": estimated_tracking_loss_ms,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    label_summary["ratio_of_total_duration"] = (
        label_summary["duration_ms"] / total_duration_ms
        if total_duration_ms
        else np.nan
    )
    label_summary["ratio_of_valid_gaze_duration"] = np.where(
        label_summary["aoi_label"].eq("InvalidGaze"),
        np.nan,
        (
            label_summary["duration_ms"] / valid_gaze_duration_ms
            if valid_gaze_duration_ms
            else np.nan
        ),
    )
    return metrics_df, label_summary


def clip_samples_to_interval(samples: pd.DataFrame, start_ms: float, end_ms: float) -> pd.DataFrame:
    df = samples[samples["gaze_time_ms"].notna()].copy()
    interval_duration_ms = max(0.0, float(end_ms) - float(start_ms))
    if df.empty:
        df.attrs["interval_duration_ms"] = interval_duration_ms
        return df
    start = pd.to_numeric(df["gaze_time_ms"], errors="coerce")
    end = start + pd.to_numeric(df["dt_ms"], errors="coerce")
    overlap = np.minimum(end, float(end_ms)) - np.maximum(start, float(start_ms))
    mask = overlap > 0
    out = df.loc[mask].copy()
    out["dt_ms"] = overlap.loc[mask].astype(float)
    out.attrs["interval_duration_ms"] = interval_duration_ms
    return out


def compute_toi_metrics(samples: pd.DataFrame, toi_path: Path, out_dir: Path, output_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, Path, Path, Path | None]:
    print("\n[4/5] Compute TOI metrics")
    toi = pd.read_csv(toi_path, encoding="utf-8-sig")

    if {"start_video_ms", "end_video_ms"}.issubset(toi.columns):
        toi["start_ms"] = pd.to_numeric(toi["start_video_ms"], errors="coerce")
        toi["end_ms"] = pd.to_numeric(toi["end_video_ms"], errors="coerce")

    required = {"start_ms", "end_ms"}
    missing = required - set(toi.columns)
    if missing:
        raise ValueError(
            f"TOI file missing columns: {missing}. Required either "
            "start_video_ms/end_video_ms or start_ms/end_ms."
        )

    metric_rows: list[pd.DataFrame] = []
    label_rows: list[pd.DataFrame] = []

    for idx, row in toi.iterrows():
        start_ms = pd.to_numeric(row.get("start_ms"), errors="coerce")
        end_ms = pd.to_numeric(row.get("end_ms"), errors="coerce")
        if pd.isna(start_ms) or pd.isna(end_ms) or float(end_ms) <= float(start_ms):
            continue
        seg = clip_samples_to_interval(samples, float(start_ms), float(end_ms))
        metrics_df, label_summary = compute_metrics(seg)

        # TOI files exported from D-Lab may already contain a column named
        # duration_ms. compute_metrics() also creates duration_ms in the label
        # summary, so inserting the raw metadata directly would raise:
        # ValueError: cannot insert duration_ms, already exists.
        #
        # Keep TOI metadata, but prefix any colliding field with "toi_".
        raw_meta = row.to_dict()
        raw_meta["toi_row_index"] = int(idx)
        raw_meta["toi_duration_ms"] = float(end_ms) - float(start_ms)

        meta: dict[str, Any] = {}
        occupied = set(metrics_df.columns).union(label_summary.columns)

        for original_key, value in raw_meta.items():
            key = str(original_key)

            # The calculated interval duration is authoritative. Ignore an
            # additional raw duration_ms field from the TOI export.
            if key == "duration_ms":
                continue

            safe_key = key
            if safe_key in occupied or safe_key in meta:
                safe_key = key if key.startswith("toi_") else f"toi_{key}"

            suffix = 2
            base_key = safe_key
            while safe_key in occupied or safe_key in meta:
                safe_key = f"{base_key}_{suffix}"
                suffix += 1

            meta[safe_key] = value

        # Insert in reverse so the TOI columns retain their original order.
        for k, v in reversed(list(meta.items())):
            metrics_df.insert(0, k, v)
            label_summary.insert(0, k, v)
        if "expected_primary_aoi" in meta and str(meta.get("expected_primary_aoi", "")).strip():
            expected = str(meta.get("expected_primary_aoi", "")).strip()
            if expected == "RobotMotionCorridor":
                expected = "RobotArm"
            col_name = f"{expected}_ratio_of_valid_gaze"
            metrics_df["expected_primary_aoi_ratio_of_valid_gaze"] = metrics_df[col_name] if col_name in metrics_df.columns else np.nan
            metrics_df["expected_primary_aoi_is_dominant"] = metrics_df["dominant_aoi_label"].astype(str).eq(expected)
        metric_rows.append(metrics_df)
        label_rows.append(label_summary)

    toi_metrics = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    toi_label_summary = pd.concat(label_rows, ignore_index=True) if label_rows else pd.DataFrame()

    toi_metrics = add_baseline_corrected_pupil(toi_metrics)
    delta_metrics = compute_t2_minus_t1_deltas(toi_metrics)

    out_toi_metrics = out_dir / f"{output_prefix}_toi_metrics.csv"
    out_toi_label = out_dir / f"{output_prefix}_toi_label_summary.csv"
    out_delta = out_dir / f"{output_prefix}_t2_minus_t1_deltas.csv" if delta_metrics is not None and not delta_metrics.empty else None
    write_csv(toi_metrics, out_toi_metrics)
    write_csv(toi_label_summary, out_toi_label)
    if out_delta is not None:
        write_csv(delta_metrics, out_delta)

    print(f"  saved: {out_toi_metrics}")
    print(f"  saved: {out_toi_label}")
    if out_delta is not None:
        print(f"  saved: {out_delta}")
    return toi_metrics, toi_label_summary, delta_metrics, out_toi_metrics, out_toi_label, out_delta


def add_baseline_corrected_pupil(toi_metrics: pd.DataFrame) -> pd.DataFrame:
    if toi_metrics.empty or "phase" not in toi_metrics.columns:
        return toi_metrics
    out = toi_metrics.copy()
    group_candidates = [c for c in ["subject_id", "recording_id", "trial_id"] if c in out.columns]
    if not group_candidates:
        group_candidates = ["__all__"]
        out["__all__"] = "all"
    for col in [
        "pupil_area_mean",
        "pupil_height_mean",
        "pupil_width_mean",
        "pupil_equivalent_diameter_mean_px",
    ]:
        if col not in out.columns:
            continue
        out[f"{col}_baseline_T0"] = np.nan
        out[f"{col}_delta_from_T0"] = np.nan
        out[f"{col}_pct_from_T0"] = np.nan
        for _, idxs in out.groupby(group_candidates).groups.items():
            idx_list = list(idxs)
            g = out.loc[idx_list]
            baseline_mask = g["phase"].astype(str).str.contains("T0|Baseline", case=False, regex=True)
            if not baseline_mask.any():
                continue
            baseline = pd.to_numeric(g.loc[baseline_mask, col], errors="coerce").mean()
            if pd.isna(baseline) or baseline == 0:
                continue
            out.loc[idx_list, f"{col}_baseline_T0"] = baseline
            out.loc[idx_list, f"{col}_delta_from_T0"] = pd.to_numeric(out.loc[idx_list, col], errors="coerce") - baseline
            out.loc[idx_list, f"{col}_pct_from_T0"] = (pd.to_numeric(out.loc[idx_list, col], errors="coerce") - baseline) / baseline * 100
    if "__all__" in out.columns:
        out = out.drop(columns=["__all__"])
    return out


def compute_t2_minus_t1_deltas(toi_metrics: pd.DataFrame) -> pd.DataFrame | None:
    if toi_metrics.empty or "phase" not in toi_metrics.columns:
        return None
    key_cols = [c for c in ["subject_id", "recording_id", "trial_id"] if c in toi_metrics.columns]
    if not key_cols:
        key_cols = ["__all__"]
        toi_metrics = toi_metrics.copy()
        toi_metrics["__all__"] = "all"

    metric_cols = [
        "Delivery_ratio_of_valid_gaze",
        "Task_ratio_of_valid_gaze",
        "RobotArm_ratio_of_valid_gaze",
        "Gripper_ratio_of_valid_gaze",
        "RobotNear_ratio_of_valid_gaze",
        "RobotTotal_Strict_ratio_of_valid_gaze",
        "RobotTotal_Buffered_ratio_of_valid_gaze",
        "RobotTotal_ratio_of_valid_gaze",
        "Outside_ratio_of_valid_gaze",
        "switch_frequency_per_s_valid",
        "robot_to_taskdelivery_switch_frequency_per_s_valid",
        "pupil_area_mean",
        "pupil_area_mean_delta_from_T0",
        "pupil_height_mean",
        "pupil_width_mean",
        "pupil_equivalent_diameter_mean_px",
        "pupil_equivalent_diameter_mean_px_delta_from_T0",
        "pupil_equivalent_diameter_peak95_px",
        "pupil_equivalent_diameter_slope_px_per_s",
    ]
    metric_cols = [c for c in metric_cols if c in toi_metrics.columns]
    phase_pairs = [
        ("T1_Reference", "T2_Manipulated"),
        ("T1_Motion", "T2_Motion"),
        ("T1_SafetyAssessment", "T2_SafetyAssessment"),
        ("T1_Placement", "T2_Placement"),
    ]

    rows: list[dict[str, Any]] = []
    for keys, g in toi_metrics.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base_meta = dict(zip(key_cols, keys))
        for t1_phase, t2_phase in phase_pairs:
            r1 = g[g["phase"].astype(str).eq(t1_phase)]
            r2 = g[g["phase"].astype(str).eq(t2_phase)]
            if r1.empty or r2.empty:
                continue
            r1 = r1.iloc[0]
            r2 = r2.iloc[0]
            row: dict[str, Any] = dict(base_meta)
            # Keep condition metadata from T2, because T2 is the manipulated event.
            for meta_col in ["condition", "danger_level", "motion_type"]:
                if meta_col in r2.index:
                    row[meta_col] = r2.get(meta_col)
            row["phase_pair"] = f"{t2_phase}_minus_{t1_phase}"
            row["t1_phase"] = t1_phase
            row["t2_phase"] = t2_phase
            for col in metric_cols:
                v1 = pd.to_numeric(pd.Series([r1.get(col)]), errors="coerce").iloc[0]
                v2 = pd.to_numeric(pd.Series([r2.get(col)]), errors="coerce").iloc[0]
                row[f"delta_{col}"] = v2 - v1 if pd.notna(v1) and pd.notna(v2) else np.nan
            rows.append(row)
    delta = pd.DataFrame(rows)
    if "__all__" in delta.columns:
        delta = delta.drop(columns=["__all__"])
    return delta


def draw_polygon(frame: np.ndarray, points: list[list[float]], color: tuple[int, int, int], label: str) -> None:
    pts = np.asarray(points, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts.reshape((-1, 1, 2))], color)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [pts.reshape((-1, 1, 2))], True, color, 3)
    c = pts.mean(axis=0).astype(int)
    cv2.putText(frame, label, (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def draw_circle(frame: np.ndarray, center: list[float], radius: float, color: tuple[int, int, int], label: str) -> None:
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.circle(frame, (cx, cy), r, color, 3)
    cv2.putText(frame, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def make_overlay_video(video_path: Path, aoi_df: pd.DataFrame, samples: pd.DataFrame, out_dir: Path, cfg: PipelineConfig, video_props: dict[str, float]) -> Path:
    print("\n[5/5] Make gaze + AOI overlay video")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = video_props["fps"]
    width = int(video_props["width"])
    height = int(video_props["height"])

    start_ms = cfg.overlay_start_ms
    end_ms = cfg.overlay_end_ms
    if start_ms is None:
        start_ms = max(0.0, float(samples["gaze_time_ms"].min()) if samples["gaze_time_ms"].notna().any() else 0.0)
    if end_ms is None:
        end_ms = min(float(video_props["frame_count"] / fps * 1000), float(samples["gaze_time_ms"].max()) if samples["gaze_time_ms"].notna().any() else 0.0)

    out_path = out_dir / f"{cfg.output_prefix}_gaze_aoi_overlay.mp4"
    ensure_parent(out_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(cfg.overlay_fps), (width, height))

    aoi_timestamps, aoi_lookup = build_aoi_lookup(aoi_df)
    sample_valid = samples[samples["gaze_time_ms"].notna()].copy()
    sample_valid = sample_valid[(sample_valid["gaze_time_ms"] >= start_ms) & (sample_valid["gaze_time_ms"] <= end_ms)].copy()

    colors = {
        "Delivery": (0, 255, 0),
        "Task": (255, 0, 0),
        "RobotArm": (0, 255, 255),
        "Gripper": (0, 0, 255),
    }

    frame_times = np.arange(float(start_ms), float(end_ms), 1000.0 / float(cfg.overlay_fps))
    for i, ts_ms in enumerate(frame_times):
        frame_idx = int(round(ts_ms / 1000 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        aoi_ts, _ = nearest_timestamp(float(ts_ms), aoi_timestamps, cfg.max_nearest_ms)
        if aoi_ts is not None:
            rows = aoi_lookup[int(aoi_ts)]
            for _, row in rows.iterrows():
                if not bool(row["available"]):
                    continue
                area = row["aoi_name"]
                color = colors.get(area, (255, 255, 255))
                if row["geometry_type"] == "polygon":
                    draw_polygon(frame, json.loads(row["points_json"]), color, area)
                elif row["geometry_type"] == "circle":
                    draw_circle(frame, json.loads(row["center_json"]), float(row["radius_px"]), color, area)

        # Use nearest gaze sample in a 100 ms window for visualization only.
        if not sample_valid.empty:
            idx = (sample_valid["gaze_time_ms"] - ts_ms).abs().idxmin()
            s = sample_valid.loc[idx]
            if abs(float(s["gaze_time_ms"] - ts_ms)) <= 100 and bool(s.get("valid_gaze", False)):
                gx, gy = int(round(float(s["gaze_x"]))), int(round(float(s["gaze_y"])))
                cv2.circle(frame, (gx, gy), 10, (0, 0, 255), -1)
                cv2.circle(frame, (gx, gy), 18, (255, 255, 255), 2)
                gaze_label = str(
                    s.get("aoi_label", "")
                )
                if gaze_label == ROBOT_NEAR_LABEL:
                    near_source = str(
                        s.get(
                            "robot_near_source",
                            "",
                        )
                    )
                    near_distance = pd.to_numeric(
                        pd.Series(
                            [
                                s.get(
                                    "robot_near_distance_px",
                                    np.nan,
                                )
                            ]
                        ),
                        errors="coerce",
                    ).iloc[0]
                    if pd.notna(near_distance):
                        gaze_label = (
                            f"RobotNear:{near_source} "
                            f"{float(near_distance):.0f}px"
                        )
                cv2.putText(frame, gaze_label, (gx + 18, gy - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"t={int(ts_ms)} ms", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  overlay frames written: {i + 1}")

    cap.release()
    writer.release()
    print(f"  saved: {out_path}")
    return out_path





def make_trial_overlay_videos(
    video_path: Path,
    aoi_df: pd.DataFrame,
    samples: pd.DataFrame,
    toi: pd.DataFrame,
    out_dir: Path,
    cfg: PipelineConfig,
    video_props: dict[str, float],
) -> Path:
    """
    Create one overlay video per Trial_Overall window.

    Only official trial time is rendered; processing padding is excluded.
    """
    print("\n[5/5] Make per-trial gaze + AOI overlays")
    overlay_dir = out_dir / "trial_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    fps = float(video_props["fps"])
    width = int(video_props["width"])
    height = int(video_props["height"])
    aoi_timestamps, aoi_lookup = build_aoi_lookup(aoi_df)

    colors = {
        "Delivery": (0, 255, 0),
        "Task": (255, 0, 0),
        "RobotArm": (0, 255, 255),
        "Gripper": (0, 0, 255),
    }

    overall = toi[
        toi["phase"].astype(str).eq("Trial_Overall")
    ].sort_values("trial_index")

    if cfg.overlay_trial_indices:
        requested = {
            int(value)
            for value in cfg.overlay_trial_indices
        }
        overall = overall[
            overall["trial_index"]
            .astype(int)
            .isin(requested)
        ].copy()
        print(
            "  Overlay trial filter: "
            + ",".join(
                map(str, sorted(requested))
            )
        )

    if overall.empty:
        print(
            "  No usable Trial_Overall window matched the "
            "requested overlay trial indices."
        )
        return overlay_dir

    for _, trial in overall.iterrows():
        trial_index = int(trial["trial_index"])
        task_id = int(trial["task_id"])
        trial_id = str(trial["trial_id"])
        start_ms = float(trial["start_video_ms"])
        end_ms = float(trial["end_video_ms"])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Could not open video: {video_path}"
            )

        out_path = (
            overlay_dir
            / f"{cfg.output_prefix}_Trial{trial_index:02d}_Task{task_id:02d}_overlay.mp4"
        )
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(cfg.overlay_fps),
            (width, height),
        )

        trial_samples = samples[
            samples["gaze_time_ms"].between(
                start_ms,
                end_ms,
                inclusive="both",
            )
        ].copy()

        phase_rows = toi[
            toi["trial_id"].astype(str).eq(trial_id)
            & ~toi["phase"].astype(str).eq("Trial_Overall")
        ].copy()

        frame_times = np.arange(
            start_ms,
            end_ms,
            1000.0 / float(cfg.overlay_fps),
        )

        for ts_ms in frame_times:
            frame_idx = int(round(ts_ms / 1000.0 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            aoi_ts, _ = nearest_timestamp(
                float(ts_ms),
                aoi_timestamps,
                cfg.max_nearest_ms,
            )
            if aoi_ts is not None:
                rows = aoi_lookup[int(aoi_ts)]
                for _, row in rows.iterrows():
                    if not bool(row["available"]):
                        continue
                    area = str(row["aoi_name"])
                    color = colors.get(
                        area,
                        (255, 255, 255),
                    )
                    if row["geometry_type"] == "polygon":
                        draw_polygon(
                            frame,
                            json.loads(row["points_json"]),
                            color,
                            area,
                        )
                    elif row["geometry_type"] == "circle":
                        draw_circle(
                            frame,
                            json.loads(row["center_json"]),
                            float(row["radius_px"]),
                            color,
                            area,
                        )

            if not trial_samples.empty:
                nearest_index = (
                    trial_samples["gaze_time_ms"] - ts_ms
                ).abs().idxmin()
                sample = trial_samples.loc[nearest_index]
                if (
                    abs(
                        float(
                            sample["gaze_time_ms"] - ts_ms
                        )
                    )
                    <= 100.0
                    and bool(sample.get("valid_gaze", False))
                ):
                    gx = int(round(float(sample["gaze_x"])))
                    gy = int(round(float(sample["gaze_y"])))
                    cv2.circle(
                        frame,
                        (gx, gy),
                        10,
                        (0, 0, 255),
                        -1,
                    )
                    cv2.circle(
                        frame,
                        (gx, gy),
                        18,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        str(sample.get("aoi_label", "")),
                        (gx + 18, gy - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            current_phase = ""
            active = phase_rows[
                phase_rows["start_video_ms"].le(ts_ms)
                & phase_rows["end_video_ms"].gt(ts_ms)
            ]
            if not active.empty:
                current_phase = str(active.iloc[0]["phase"])

            cv2.putText(
                frame,
                (
                    f"Trial {trial_index:02d} / Task {task_id:02d} "
                    f"/ {current_phase}"
                ),
                (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"video t={int(ts_ms)} ms",
                (24, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

        cap.release()
        writer.release()
        print(f"  saved: {out_path}")

    return overlay_dir


def run_one(cfg: PipelineConfig) -> dict[str, Any]:
    if not cfg.video.exists():
        raise FileNotFoundError(f"Video not found: {cfg.video}")
    if cfg.gaze is None or not cfg.gaze.exists():
        raise FileNotFoundError(
            f"Gaze file not found: {cfg.gaze}"
        )
    if cfg.timeline_file is None:
        raise ValueError(
            "Windowed mode requires --timeline_file."
        )
    if not cfg.timeline_file.exists():
        raise FileNotFoundError(
            f"Timeline file not found: {cfg.timeline_file}"
        )
    if cfg.participant_id is None:
        raise ValueError(
            "--participant_id is required with --timeline_file."
        )

    out_dir = cfg.output_root.expanduser().resolve() / cfg.output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_marker_config(cfg.config_path)

    # Backward-compatible role update for Gripper marker 15.
    marker_roles = config.setdefault("marker_roles", {})
    gripper_role_ids = [
        int(marker_id)
        for marker_id in marker_roles.get("Gripper", [])
    ]
    for marker_id in (13, 14, 15):
        if marker_id not in gripper_role_ids:
            gripper_role_ids.append(marker_id)
    marker_roles["Gripper"] = sorted(gripper_role_ids)

    video_props = get_video_properties(cfg.video)
    if cfg.aoi_frame_width is None:
        cfg.aoi_frame_width = video_props["width"]
    if cfg.aoi_frame_height is None:
        cfg.aoi_frame_height = video_props["height"]

    print("=" * 78)
    print("DG3 Trial-Window Dynamic-AOI Pipeline")
    print(f"Output dir: {out_dir}")
    print("=" * 78)

    # -------------------------------------------------------------
    # 0. Align the external timeline before processing video frames.
    # -------------------------------------------------------------
    alignment = resolve_alignment_from_gaze_file(
        cfg.gaze,
        video_props,
        cfg,
    )

    timeline, timeline_metadata = read_timeline_file(
        cfg.timeline_file,
        participant_id=int(cfg.participant_id),
    )

    timeline_toi, timeline_toi_csv, timeline_qa_csv = (
        convert_timeline_to_toi(
            timeline=timeline,
            alignment=alignment,
            video_props=video_props,
            cfg=cfg,
            out_dir=out_dir,
        )
    )

    processing_windows, processing_windows_csv = (
        build_trial_processing_windows(
            timeline_toi,
            video_props,
            cfg,
            out_dir,
        )
    )

    # -------------------------------------------------------------
    # 1. Process only trial video frames.
    # -------------------------------------------------------------
    if cfg.marker_detections:
        detections_csv = cfg.marker_detections
        detections_df = pd.read_csv(
            detections_csv,
            encoding="utf-8-sig",
        )
        if cfg.marker_frame_summary is None:
            raise ValueError(
                "When --marker_detections is used, also supply "
                "--marker_frame_summary."
            )
        frame_summary_csv = cfg.marker_frame_summary
        frame_summary_df = pd.read_csv(
            frame_summary_csv,
            encoding="utf-8-sig",
        )
        marker_rates_csv = None
        print(
            f"\n[1/5] Using existing windowed detections: "
            f"{detections_csv}"
        )
    else:
        (
            detections_df,
            frame_summary_df,
            _rates_df,
            detections_csv,
            frame_summary_csv,
            marker_rates_csv,
            video_props,
        ) = detect_markers_from_video_windows(
            cfg.video,
            config,
            out_dir,
            cfg,
            processing_windows,
        )

    # -------------------------------------------------------------
    # 2. Build AOIs independently per trial.
    # -------------------------------------------------------------
    if cfg.dynamic_aoi:
        dynamic_aoi_csv = cfg.dynamic_aoi
        aoi_df = pd.read_csv(
            dynamic_aoi_csv,
            encoding="utf-8-sig",
        )
        dynamic_summary_csv = None
        availability_csv = None
        trial_availability_csv = None
        print(
            f"\n[2/5] Using existing windowed dynamic AOI: "
            f"{dynamic_aoi_csv}"
        )
    else:
        (
            aoi_df,
            _aoi_summary_df,
            _overall_availability_df,
            dynamic_aoi_csv,
            dynamic_summary_csv,
            availability_csv,
            trial_availability_csv,
        ) = build_dynamic_aoi_per_trial(
            detections_df,
            frame_summary_df,
            config,
            out_dir,
            cfg,
            processing_windows,
        )

    # -------------------------------------------------------------
    # 3. Keep only gaze rows inside padded trial windows.
    # -------------------------------------------------------------
    (
        samples,
        _overall_metrics,
        _label_summary,
        samples_csv,
        overall_metrics_csv,
        label_summary_csv,
        diagnostics,
    ) = classify_gaze(
        cfg.gaze,
        aoi_df,
        out_dir,
        cfg,
        video_props,
        processing_windows=processing_windows,
        precomputed_alignment=alignment,
    )

    samples = annotate_samples_with_toi(
        samples,
        timeline_toi,
    )
    write_csv(samples, samples_csv)

    # -------------------------------------------------------------
    # 4. Compute trial, T0, T1, T2 metrics using unchanged formulas.
    # -------------------------------------------------------------
    (
        _toi_metrics,
        _toi_label_summary,
        _delta_metrics,
        toi_metrics_csv,
        toi_label_csv,
        delta_csv,
    ) = compute_toi_metrics(
        samples,
        timeline_toi_csv,
        out_dir,
        cfg.output_prefix,
    )

    trial_clips_dir = None
    if cfg.export_trial_clips:
        trial_clips_dir = export_raw_trial_clips(
            video_path=cfg.video,
            toi=timeline_toi,
            out_dir=out_dir,
            cfg=cfg,
            video_props=video_props,
        )

    # -------------------------------------------------------------
    # 5. Trial overlays only; no full-session gap rendering.
    # -------------------------------------------------------------
    overlay_dir = None
    if cfg.make_overlay:
        overlay_dir = make_trial_overlay_videos(
            cfg.video,
            aoi_df,
            samples,
            timeline_toi,
            out_dir,
            cfg,
            video_props,
        )
    else:
        print(
            "\n[5/5] Trial overlays skipped. "
            "Add --make_overlay if needed."
        )

    video_duration_ms = (
        float(video_props["frame_count"])
        / float(video_props["fps"])
        * 1000.0
    )
    processed_duration_ms = float(
        processing_windows["processing_duration_ms"].sum()
    )

    summary = {
        "output_prefix": cfg.output_prefix,
        "mode": "timeline_windowed",
        "video": str(cfg.video),
        "gaze": str(cfg.gaze),
        "timeline_file": str(cfg.timeline_file),
        "participant_id": cfg.participant_id,
        "output_dir": str(out_dir),
        "timeline_toi_csv": str(timeline_toi_csv),
        "timeline_alignment_qa_csv": str(
            timeline_qa_csv
        ),
        "processing_windows_csv": str(
            processing_windows_csv
        ),
        "full_video_duration_ms": video_duration_ms,
        "processed_window_duration_ms": processed_duration_ms,
        "processed_video_fraction": (
            processed_duration_ms / video_duration_ms
            if video_duration_ms > 0
            else np.nan
        ),
        "marker_detections_csv": str(detections_csv),
        "marker_frame_summary_csv": str(frame_summary_csv),
        "marker_detection_rates_csv": str(
            marker_rates_csv or ""
        ),
        "dynamic_aoi_csv": str(dynamic_aoi_csv),
        "dynamic_aoi_summary_csv": str(
            dynamic_summary_csv or ""
        ),
        "aoi_availability_csv": str(
            availability_csv or ""
        ),
        "trial_aoi_availability_csv": str(
            trial_availability_csv or ""
        ),
        "samples_csv": str(samples_csv),
        "overall_metrics_csv": str(
            overall_metrics_csv
        ),
        "label_summary_csv": str(label_summary_csv),
        "toi_metrics_csv": str(toi_metrics_csv),
        "toi_label_summary_csv": str(toi_label_csv),
        "t2_minus_t1_delta_csv": str(
            delta_csv or ""
        ),
        "trial_clips_dir": str(
            trial_clips_dir or ""
        ),
        "trial_overlay_dir": str(
            overlay_dir or ""
        ),
        **diagnostics,
    }

    summary_path = (
        out_dir / f"{cfg.output_prefix}_run_summary.csv"
    )
    write_csv(pd.DataFrame([summary]), summary_path)

    print("\nDone.")
    print(f"Run summary: {summary_path}")
    return summary


def parse_optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "" or str(value).strip().lower() == "auto":
        return None
    return float(value)



def parse_trial_index_list(
    value: str | None,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    token = str(value).strip()
    if token == "" or token.lower() in {
        "all",
        "*",
        "none",
    }:
        return None

    result: list[int] = []
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start_value = int(start_text)
            end_value = int(end_text)
            if end_value < start_value:
                start_value, end_value = (
                    end_value,
                    start_value,
                )
            result.extend(
                range(start_value, end_value + 1)
            )
        else:
            result.append(int(part))

    unique = tuple(sorted(set(result)))
    return unique or None


def cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        video=Path(args.video),
        gaze=Path(args.gaze) if args.gaze else None,
        output_prefix=args.output_prefix,
        output_root=Path(args.output_root),
        toi_csv=Path(args.toi_csv) if args.toi_csv else None,
        triggered_tasks_txt=None,
        auto_find_triggered_tasks=False,
        timeline_file=(
            Path(args.timeline_file)
            if args.timeline_file
            else None
        ),
        participant_id=(
            int(args.participant_id)
            if args.participant_id is not None
            else None
        ),
        timeline_clock_offset_ms=float(
            args.timeline_clock_offset_ms
        ),
        timeline_clock_drift_ppm=float(
            args.timeline_clock_drift_ppm
        ),
        export_trial_clips=bool(args.export_trial_clips),
        clip_padding_before_ms=float(
            args.clip_padding_before_ms
        ),
        clip_padding_after_ms=float(
            args.clip_padding_after_ms
        ),
        trial_processing_padding_before_ms=float(
            args.trial_processing_padding_before_ms
        ),
        trial_processing_padding_after_ms=float(
            args.trial_processing_padding_after_ms
        ),
        video_start_rec_ms=parse_optional_float(args.video_start_rec_ms),
        config_path=Path(args.config),
        sample_every_sec=float(args.sample_every_sec),
        max_nearest_ms=float(args.max_nearest_ms),
        gaze_continuity_gap_ms=float(args.gaze_continuity_gap_ms),
        time_offset_ms=float(args.time_offset_ms),
        gaze_native_width=float(args.gaze_native_width),
        gaze_native_height=float(args.gaze_native_height),
        aoi_frame_width=parse_optional_float(args.aoi_frame_width),
        aoi_frame_height=parse_optional_float(args.aoi_frame_height),
        disable_scaling=bool(args.disable_scaling),
        delivery_scale=float(args.delivery_scale),
        task_scale=float(args.task_scale),
        static_hold_last_ms=float(args.static_hold_last_ms),
        robot_arm_marker_radius_px=float(args.robot_arm_marker_radius_px),
        robot_hold_last_ms=float(args.robot_hold_last_ms),
        gripper_base_radius_px=float(args.gripper_base_radius_px),
        gripper_single_marker_radius_px=float(args.gripper_single_marker_radius_px),
        gripper_single_marker_major_radius_px=float(args.gripper_single_marker_major_radius_px),
        gripper_single_marker_minor_radius_px=float(args.gripper_single_marker_minor_radius_px),
        gripper_single_marker_angle_offset_deg=float(args.gripper_single_marker_angle_offset_deg),
        table_outer_expansion_px=float(
            args.table_outer_expansion_px
        ),
        table_ray_bottom_margin_px=float(
            args.table_ray_bottom_margin_px
        ),
        robot_near_enabled=not bool(
            args.disable_robot_near
        ),
        robot_near_gripper_margin_px=float(
            args.robot_near_gripper_margin_px
        ),
        robot_near_arm_margin_px=float(
            args.robot_near_arm_margin_px
        ),
        gripper_max_radius_px=float(args.gripper_max_radius_px),
        gripper_hold_last_ms=float(args.gripper_hold_last_ms),
        detector_min_perimeter_rate=float(args.detector_min_perimeter_rate),
        optical_flow_enabled=not bool(args.disable_optical_flow),
        optical_flow_max_age_ms=float(args.optical_flow_max_age_ms),
        table_optical_flow_max_age_ms=float(args.table_optical_flow_max_age_ms),
        table_validated_hold_enabled=not bool(args.disable_table_validated_hold),
        table_validated_hold_max_frames=int(args.table_validated_hold_max_frames),
        table_validated_hold_max_shift_px=float(args.table_validated_hold_max_shift_px),
        table_ransac_reproj_threshold_px=float(args.table_ransac_reproj_threshold_px),
        table_reacquire_min_direct_markers=int(args.table_reacquire_min_direct_markers),
        table_reacquire_consecutive_frames=int(args.table_reacquire_consecutive_frames),
        table_corner_reacquire_consecutive_frames=int(args.table_corner_reacquire_consecutive_frames),
        table_corner_template_min_calibration_markers=int(args.table_corner_template_min_calibration_markers),
        table_corner_template_update_alpha=float(args.table_corner_template_update_alpha),
        table_corner_min_pair_distance_norm=float(args.table_corner_min_pair_distance_norm),
        table_corner_ransac_reproj_threshold_px=float(args.table_corner_ransac_reproj_threshold_px),
        table_corner_max_reprojection_error_px=float(args.table_corner_max_reprojection_error_px),
        table_min_direct_markers_while_locked=int(args.table_min_direct_markers_while_locked),
        table_flow_only_bridge_max_age_ms=float(args.table_flow_only_bridge_max_age_ms),
        scene_motion_reset_enabled=not bool(args.disable_scene_motion_reset),
        scene_motion_max_features=int(args.scene_motion_max_features),
        scene_motion_min_valid_points=int(args.scene_motion_min_valid_points),
        scene_motion_fb_error_px=float(args.scene_motion_fb_error_px),
        scene_motion_median_threshold_px=float(args.scene_motion_median_threshold_px),
        scene_motion_p90_threshold_px=float(args.scene_motion_p90_threshold_px),
        scene_motion_min_valid_fraction=float(args.scene_motion_min_valid_fraction),
        optical_flow_fb_error_px=float(args.optical_flow_fb_error_px),
        optical_flow_min_valid_corners=int(args.optical_flow_min_valid_corners),
        optical_flow_win_size_px=int(args.optical_flow_win_size_px),
        optical_flow_max_level=int(args.optical_flow_max_level),
        optical_flow_max_center_step_px=float(args.optical_flow_max_center_step_px),
        optical_flow_min_area_ratio=float(args.optical_flow_min_area_ratio),
        optical_flow_max_area_ratio=float(args.optical_flow_max_area_ratio),
        table_smoothing_alpha=float(args.table_smoothing_alpha),
        robot_smoothing_alpha=float(args.robot_smoothing_alpha),
        gripper_smoothing_alpha=float(args.gripper_smoothing_alpha),
        table_max_reprojection_error_px=float(args.table_max_reprojection_error_px),
        table_max_centroid_jump_px=float(args.table_max_centroid_jump_px),
        table_min_area_ratio=float(args.table_min_area_ratio),
        table_max_area_ratio=float(args.table_max_area_ratio),
        table_min_polygon_area_px2=float(args.table_min_polygon_area_px2),
        temporal_quality_window_ms=float(args.temporal_quality_window_ms),
        make_overlay=bool(args.make_overlay),
        overlay_start_ms=parse_optional_float(args.overlay_start_ms),
        overlay_end_ms=parse_optional_float(args.overlay_end_ms),
        overlay_fps=float(args.overlay_fps),
        overlay_trial_indices=parse_trial_index_list(
            args.overlay_trial_indices
        ),
        require_full_phase_video_coverage=not bool(
            args.allow_partial_video_phases
        ),
        video_coverage_tolerance_ms=float(
            args.video_coverage_tolerance_ms
        ),
        time_col=args.time_col,
        gaze_x_col=args.gaze_x_col,
        gaze_y_col=args.gaze_y_col,
        marker_detections=Path(args.marker_detections) if args.marker_detections else None,
        marker_frame_summary=Path(args.marker_frame_summary) if args.marker_frame_summary else None,
        dynamic_aoi=Path(args.dynamic_aoi) if args.dynamic_aoi else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "DG3 trial-window pipeline: align an external timeline, detect "
            "ArUco markers, construct dynamic AOIs, classify gaze, and export "
            "trial/phase metrics."
        )
    )
    p.add_argument(
        "--video",
        required=True,
        help="Path to the concatenated DG3 scene-camera video.",
    )
    p.add_argument(
        "--gaze",
        required=True,
        help="Path to the D-Lab gaze export (TXT, CSV, or TSV).",
    )
    p.add_argument("--toi_csv", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--timeline_file",
        required=True,
        help=(
            "Fine-phase timeline exported by the experiment computer. "
            "Preferred format: .tsv. Also supports .txt, .csv, .xlsx, "
            "and the older .numbers format."
        ),
    )
    p.add_argument(
        "--participant_id",
        type=int,
        required=True,
        help=(
            "Numeric user_id to select from the timeline file, "
            "for example 3 for participant ID3."
        ),
    )
    p.add_argument(
        "--timeline_clock_offset_ms",
        type=float,
        default=0.0,
        help=(
            "Additive correction applied to timeline UTC timestamps "
            "before mapping them to DG3 recording time. Default: 0 ms."
        ),
    )
    p.add_argument(
        "--timeline_clock_drift_ppm",
        type=float,
        default=0.0,
        help=(
            "Linear clock-drift correction around the first timeline "
            "timestamp. Default: 0 ppm."
        ),
    )
    p.add_argument(
        "--export_trial_clips",
        action="store_true",
        help="Export one raw video clip for every Trial_Overall interval.",
    )
    p.add_argument(
        "--clip_padding_before_ms",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--clip_padding_after_ms",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--trial_processing_padding_before_ms",
        type=float,
        default=1000.0,
        help=(
            "Extra video/gaze context processed before each trial. "
            "Metrics exclude this padding. Default: 1000 ms."
        ),
    )
    p.add_argument(
        "--trial_processing_padding_after_ms",
        type=float,
        default=500.0,
        help=(
            "Extra video/gaze context processed after each trial. "
            "Metrics exclude this padding. Default: 500 ms."
        ),
    )
    p.add_argument(
        "--video_start_rec_ms",
        default="auto",
        help=(
            "D-Lab rec_time corresponding to video frame 0. Use 'auto' or "
            "a numeric millisecond value."
        ),
    )
    p.add_argument(
        "--output_prefix",
        default="dg3_analysis",
        help="Participant/run identifier used for the output folder and files.",
    )
    p.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=(
            "Directory that receives one subfolder per output prefix. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Marker configuration JSON. Default: {DEFAULT_CONFIG}",
    )
    p.add_argument("--sample_every_sec", type=float, default=0.033)
    p.add_argument("--detector_min_perimeter_rate", type=float, default=0.012)
    p.add_argument("--max_nearest_ms", type=float, default=80)
    p.add_argument(
        "--gaze_continuity_gap_ms",
        type=float,
        default=100,
        help=(
            "Maximum gap between valid scene-gaze samples treated as "
            "continuous coverage. Default: 100 ms."
        ),
    )
    p.add_argument(
        "--time_offset_ms",
        type=float,
        default=0.0,
        help=(
            "Legacy alignment option. A value of -61762 is interpreted as "
            "video_start_rec_ms=61762; original rec_time and UTC remain unchanged."
        ),
    )
    p.add_argument("--gaze_native_width", type=float, default=2048)
    p.add_argument("--gaze_native_height", type=float, default=1152)
    p.add_argument("--aoi_frame_width", default="auto")
    p.add_argument("--aoi_frame_height", default="auto")
    p.add_argument("--disable_scaling", action="store_true")

    p.add_argument("--delivery_scale", type=float, default=1.05)
    p.add_argument("--task_scale", type=float, default=1.05)
    p.add_argument("--static_hold_last_ms", type=float, default=0, help="Legacy option; keep at 0 in v5.")
    p.add_argument(
        "--robot_arm_marker_radius_px",
        type=float,
        default=110,
        help="Radius of each marker-centred RobotArm circle. Default: 110 px.",
    )
    p.add_argument("--robot_hold_last_ms", type=float, default=0, help="Legacy option; keep at 0 in v5.")
    p.add_argument("--gripper_base_radius_px", type=float, default=70)
    p.add_argument(
        "--gripper_single_marker_radius_px",
        type=float,
        default=125,
        help=(
            "Envelope thickness for connected Gripper ellipses made from "
            "2–3 visible markers. Default: 125 px."
        ),
    )
    p.add_argument(
        "--gripper_single_marker_major_radius_px",
        type=float,
        default=150,
        help=(
            "Major semi-axis of a single-marker Gripper ellipse. "
            "Default: 150 px."
        ),
    )
    p.add_argument(
        "--gripper_single_marker_minor_radius_px",
        type=float,
        default=110,
        help=(
            "Minor semi-axis of a single-marker Gripper ellipse. "
            "Default: 110 px."
        ),
    )
    p.add_argument(
        "--gripper_single_marker_angle_offset_deg",
        type=float,
        default=90,
        help=(
            "Rotation offset added to the marker local x-axis for a "
            "single-marker Gripper ellipse. Use 90 if the marker top edge "
            "is perpendicular to the gripper long axis. Default: 90 degrees."
        ),
    )
    p.add_argument(
        "--table_outer_expansion_px",
        type=float,
        default=35.0,
        help=(
            "Fallback expansion used only when a table region has exactly "
            "three usable markers and is represented by a direct triangle. "
            "Default: 35 px."
        ),
    )
    p.add_argument(
        "--table_ray_bottom_margin_px",
        type=float,
        default=0.0,
        help=(
            "Distance above the image bottom where Task/Delivery boundary "
            "rays are clipped. Default: 0 px, i.e. the frame bottom."
        ),
    )
    p.add_argument(
        "--robot_near_gripper_margin_px",
        type=float,
        default=60.0,
        help=(
            "Maximum outside distance from the Gripper AOI that can be "
            "classified as RobotNear. Applied only to gaze otherwise "
            "classified as Outside. Default: 60 px."
        ),
    )
    p.add_argument(
        "--robot_near_arm_margin_px",
        type=float,
        default=40.0,
        help=(
            "Maximum outside distance from the RobotArm AOI that can be "
            "classified as RobotNear. Applied only to gaze otherwise "
            "classified as Outside. Default: 40 px."
        ),
    )
    p.add_argument(
        "--disable_robot_near",
        action="store_true",
        help=(
            "Disable the RobotNear tolerance and use strict AOI "
            "classification only."
        ),
    )
    p.add_argument("--gripper_max_radius_px", type=float, default=120)
    p.add_argument("--gripper_hold_last_ms", type=float, default=0, help="Legacy option; keep at 0 in v5.")
    p.add_argument("--robot_scale", type=float, default=1.05, help=argparse.SUPPRESS)

    p.add_argument("--disable_optical_flow", action="store_true")
    p.add_argument("--optical_flow_max_age_ms", type=float, default=150)
    p.add_argument(
        "--table_optical_flow_max_age_ms",
        type=float,
        default=250,
        help=(
            "Maximum table optical-flow bridge age. "
            "Default: 250 ms."
        ),
    )
    p.add_argument(
        "--disable_table_validated_hold",
        action="store_true",
        help=(
            "Disable the offline forward-backward validated short "
            "table-marker hold."
        ),
    )
    p.add_argument(
        "--table_validated_hold_max_frames",
        type=int,
        default=3,
        help=(
            "Maximum number of missing sampled frames filled from the "
            "previous marker position after post-gap validation. Default: 3."
        ),
    )
    p.add_argument(
        "--table_validated_hold_max_shift_px",
        type=float,
        default=45.0,
        help=(
            "Maximum pre/post marker-centre displacement allowed for a "
            "validated hold. Larger shifts keep the triangle fallback. "
            "Default: 45 px."
        ),
    )
    p.add_argument(
        "--table_reacquire_min_direct_markers",
        type=int,
        default=3,
        help="Direct table markers required to reacquire after a head turn.",
    )
    p.add_argument(
        "--table_reacquire_consecutive_frames",
        type=int,
        default=1,
        help=(
            "Consecutive >=3-marker direct-anchor frames required for "
            "high-confidence reacquisition. Default: 1."
        ),
    )
    p.add_argument(
        "--table_corner_reacquire_consecutive_frames",
        type=int,
        default=2,
        help=(
            "Consecutive valid two-marker corner-homography frames "
            "required for medium-confidence reacquisition. Default: 2."
        ),
    )
    p.add_argument(
        "--table_corner_template_min_calibration_markers",
        type=int,
        default=4,
        help=(
            "Direct markers required to learn/update marker-corner "
            "templates. Default: 4."
        ),
    )
    p.add_argument(
        "--table_corner_template_update_alpha",
        type=float,
        default=0.15,
        help="EMA alpha for learned normalized marker-corner templates.",
    )
    p.add_argument(
        "--table_corner_min_pair_distance_norm",
        type=float,
        default=0.45,
        help=(
            "Minimum normalized distance between the two direct markers "
            "used for corner-based reacquisition."
        ),
    )
    p.add_argument(
        "--table_corner_ransac_reproj_threshold_px",
        type=float,
        default=7.0,
        help="RANSAC point-inlier threshold for corner homography.",
    )
    p.add_argument(
        "--table_corner_max_reprojection_error_px",
        type=float,
        default=9.0,
        help="Maximum RMS corner reprojection error.",
    )
    p.add_argument(
        "--table_min_direct_markers_while_locked",
        type=int,
        default=2,
        help="Minimum direct table markers for a mixed short bridge.",
    )
    p.add_argument(
        "--table_flow_only_bridge_max_age_ms",
        type=float,
        default=120,
        help="Maximum flow-only table bridge after the last direct anchor.",
    )
    p.add_argument(
        "--disable_scene_motion_reset",
        action="store_true",
        help="Disable global head-turn/scene-motion reset (not recommended).",
    )
    p.add_argument("--scene_motion_max_features", type=int, default=250)
    p.add_argument("--scene_motion_min_valid_points", type=int, default=24)
    p.add_argument("--scene_motion_fb_error_px", type=float, default=2.0)
    p.add_argument("--scene_motion_median_threshold_px", type=float, default=22.0)
    p.add_argument("--scene_motion_p90_threshold_px", type=float, default=55.0)
    p.add_argument("--scene_motion_min_valid_fraction", type=float, default=0.30)
    p.add_argument(
        "--table_ransac_reproj_threshold_px",
        type=float,
        default=10.0,
        help="RANSAC inlier threshold for tabletop mapping. Default: 10 px.",
    )
    p.add_argument("--optical_flow_fb_error_px", type=float, default=1.5)
    p.add_argument("--optical_flow_min_valid_corners", type=int, default=3)
    p.add_argument("--optical_flow_win_size_px", type=int, default=31)
    p.add_argument("--optical_flow_max_level", type=int, default=3)
    p.add_argument("--optical_flow_max_center_step_px", type=float, default=140)
    p.add_argument("--optical_flow_min_area_ratio", type=float, default=0.45)
    p.add_argument("--optical_flow_max_area_ratio", type=float, default=2.20)

    p.add_argument("--table_smoothing_alpha", type=float, default=0.65)
    p.add_argument("--robot_smoothing_alpha", type=float, default=0.75)
    p.add_argument("--gripper_smoothing_alpha", type=float, default=0.75)
    p.add_argument("--table_max_reprojection_error_px", type=float, default=12.0)
    p.add_argument("--table_max_centroid_jump_px", type=float, default=260.0)
    p.add_argument("--table_min_area_ratio", type=float, default=0.40)
    p.add_argument("--table_max_area_ratio", type=float, default=2.50)
    p.add_argument("--table_min_polygon_area_px2", type=float, default=800.0)
    p.add_argument("--temporal_quality_window_ms", type=float, default=500.0)

    p.add_argument("--make_overlay", action="store_true")
    p.add_argument("--overlay_start_ms", default=None)
    p.add_argument("--overlay_end_ms", default=None)
    p.add_argument("--overlay_fps", type=float, default=10)
    p.add_argument(
        "--overlay_trial_indices",
        default=None,
        help=(
            "Comma-separated Trial_Overall indices to render, e.g. "
            "'3' or '2,5,7'. Omit or use 'all' to render all usable trials."
        ),
    )
    p.add_argument(
        "--video_coverage_tolerance_ms",
        type=float,
        default=250.0,
        help=(
            "Small boundary tolerance when checking whether a complete "
            "phase is covered by the available video. Default: 250 ms."
        ),
    )
    p.add_argument(
        "--allow_partial_video_phases",
        action="store_true",
        help=(
            "Allow phase intervals that extend outside the available video. "
            "Not recommended for formal analysis."
        ),
    )
    p.add_argument("--time_col", default=None)
    p.add_argument("--gaze_x_col", default=None)
    p.add_argument("--gaze_y_col", default=None)
    p.add_argument("--marker_detections", default=None)
    p.add_argument("--marker_frame_summary", default=None)
    p.add_argument("--dynamic_aoi", default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    require_runtime_dependencies()
    cfg = cfg_from_args(args)
    run_one(cfg)


if __name__ == "__main__":
    main()
