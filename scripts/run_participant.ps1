[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^P\d+$")]
    [string]$ParticipantId,

    [string]$ProjectRoot = "",
    [string]$VideoPath = "",
    [string]$GazePath = "",
    [string]$TimelineFile = "",
    [string]$PythonExe = "",
    [string]$ConfigFile = "",
    [string]$OutputRoot = "",
    [string]$ReuseMarkerDir = "",
    [string]$OverlayTrialIndices = "",
    [switch]$NoOverlay,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
elseif (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw ("Project root not found: " + $ProjectRoot)
}
else {
    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
}

$SessionFolder = Join-Path $ProjectRoot ("data_raw\" + $ParticipantId)
$Pipeline = Join-Path $PSScriptRoot "hri_pipeline.py"

if ([string]::IsNullOrWhiteSpace($ConfigFile)) {
    $ConfigFile = Join-Path $ProjectRoot "metadata\marker_config_shared6.json"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "outputs"
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $WindowsVenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $PosixVenvPython = Join-Path $ProjectRoot ".venv\bin\python"

    if (Test-Path -LiteralPath $WindowsVenvPython -PathType Leaf) {
        $PythonExe = $WindowsVenvPython
    }
    elseif (Test-Path -LiteralPath $PosixVenvPython -PathType Leaf) {
        $PythonExe = $PosixVenvPython
    }
    else {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $PythonCommand) {
            throw "Python was not found. Create .venv or pass -PythonExe."
        }
        $PythonExe = $PythonCommand.Source
    }
}

foreach ($RequiredPath in @($SessionFolder, $Pipeline, $ConfigFile)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw ("Required path not found: " + $RequiredPath)
    }
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    $PythonCommand = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if ($null -eq $PythonCommand) {
        throw ("Python executable not found: " + $PythonExe)
    }
    $PythonExe = $PythonCommand.Source
}

$NumericParticipantId = [int]([regex]::Match($ParticipantId, "\d+").Value)
$Files = @(Get-ChildItem -LiteralPath $SessionFolder -File)

if ([string]::IsNullOrWhiteSpace($VideoPath)) {
    $Videos = @(
        $Files | Where-Object {
            @(".mp4", ".avi", ".mov", ".mkv") -contains $_.Extension.ToLowerInvariant()
        }
    )
    if ($Videos.Count -ne 1) {
        throw (
            "Expected exactly one scene video in " + $SessionFolder +
            ", found " + [string]$Videos.Count +
            ". Pass -VideoPath when the folder contains multiple videos."
        )
    }
    $VideoPath = $Videos[0].FullName
}

if ([string]::IsNullOrWhiteSpace($GazePath)) {
    $Gazes = @(
        $Files | Where-Object {
            ($_.Extension.ToLowerInvariant() -in @(".txt", ".csv", ".tsv")) -and
            ($_.Name -match "(?i)gaze|eye.?data|csvdata")
        }
    )
    if ($Gazes.Count -ne 1) {
        throw (
            "Expected exactly one gaze export in " + $SessionFolder +
            ", found " + [string]$Gazes.Count +
            ". Pass -GazePath when automatic detection is ambiguous."
        )
    }
    $GazePath = $Gazes[0].FullName
}

if ([string]::IsNullOrWhiteSpace($TimelineFile)) {
    $TimelineFile = Join-Path $SessionFolder "timeline.tsv"
}

foreach ($InputPath in @($VideoPath, $GazePath, $TimelineFile)) {
    if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
        throw ("Input file not found: " + $InputPath)
    }
}

$Arguments = @(
    $Pipeline,
    "--video", $VideoPath,
    "--gaze", $GazePath,
    "--timeline_file", $TimelineFile,
    "--participant_id", [string]$NumericParticipantId,
    "--output_root", $OutputRoot,
    "--output_prefix", $ParticipantId,
    "--config", $ConfigFile,
    "--trial_processing_padding_before_ms", "1000",
    "--trial_processing_padding_after_ms", "500",
    "--video_coverage_tolerance_ms", "250",
    "--gaze_native_width", "2048",
    "--gaze_native_height", "1152",
    "--sample_every_sec", "0.033",
    "--max_nearest_ms", "80",
    "--optical_flow_max_age_ms", "150",
    "--table_optical_flow_max_age_ms", "180",
    "--table_flow_only_bridge_max_age_ms", "100",
    "--table_validated_hold_max_frames", "6",
    "--table_validated_hold_max_shift_px", "45",
    "--static_hold_last_ms", "0",
    "--robot_hold_last_ms", "0",
    "--gripper_hold_last_ms", "0",
    "--robot_arm_marker_radius_px", "100",
    "--table_outer_expansion_px", "35",
    "--table_ray_bottom_margin_px", "0",
    "--gripper_single_marker_radius_px", "125",
    "--gripper_single_marker_major_radius_px", "150",
    "--gripper_single_marker_minor_radius_px", "110",
    "--gripper_single_marker_angle_offset_deg", "90",
    "--robot_near_gripper_margin_px", "60",
    "--robot_near_arm_margin_px", "40"
)

if (-not [string]::IsNullOrWhiteSpace($ReuseMarkerDir)) {
    $DetectionFile = Join-Path $ReuseMarkerDir ($ParticipantId + "_windowed_marker_detections.csv")
    $FrameSummaryFile = Join-Path $ReuseMarkerDir ($ParticipantId + "_windowed_marker_frame_summary.csv")

    if (
        (Test-Path -LiteralPath $DetectionFile -PathType Leaf) -and
        (Test-Path -LiteralPath $FrameSummaryFile -PathType Leaf)
    ) {
        $Arguments += @(
            "--marker_detections", $DetectionFile,
            "--marker_frame_summary", $FrameSummaryFile
        )
    }
    else {
        throw ("Reusable marker files were not found in: " + $ReuseMarkerDir)
    }
}

if (-not $NoOverlay) {
    $Arguments += @("--make_overlay", "--overlay_fps", "5")

    if (-not [string]::IsNullOrWhiteSpace($OverlayTrialIndices)) {
        $Arguments += @("--overlay_trial_indices", $OverlayTrialIndices)
    }
}

if ($DryRun) {
    $DisplayArguments = $Arguments | ForEach-Object {
        '"' + ([string]$_).Replace('"', '\"') + '"'
    }
    Write-Host ($PythonExe + " " + ($DisplayArguments -join " "))
    return
}

& $PythonExe @Arguments

if ($LASTEXITCODE -ne 0) {
    throw ("Pipeline failed with exit code " + [string]$LASTEXITCODE)
}

Write-Host ""
Write-Host "Participant completed." -ForegroundColor Green
Write-Host ("Output: " + (Join-Path $OutputRoot $ParticipantId))
