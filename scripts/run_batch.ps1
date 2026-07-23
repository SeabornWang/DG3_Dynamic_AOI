[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 9999)]
    [int]$StartId,

    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 9999)]
    [int]$EndId,

    [string]$ProjectRoot = "",
    [string]$OutputRoot = "",
    [string]$PythonExe = "",
    [string]$ConfigFile = "",
    [int]$MaxParallel = 2,
    [ValidateSet("Metrics", "Overlay", "All")]
    [string]$Stage = "Metrics",
    [string]$TimelineName = "timeline.tsv",
    [double]$LowAvailabilityThreshold = 0.60,
    [int]$RandomSeed = 20260726,
    [switch]$ReuseExistingMarkers,
    [string]$ReuseRoot = ""
)

$ErrorActionPreference = "Stop"

if ($MaxParallel -lt 1) {
    throw "MaxParallel must be at least 1."
}

if ($EndId -lt $StartId) {
    throw "EndId must be greater than or equal to StartId."
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
elseif (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw ("Project root not found: " + $ProjectRoot)
}
else {
    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "outputs"
}

if ($ReuseExistingMarkers -and [string]::IsNullOrWhiteSpace($ReuseRoot)) {
    throw "Pass -ReuseRoot when -ReuseExistingMarkers is enabled."
}

$Runner = Join-Path $PSScriptRoot "run_participant.ps1"
$PowerShellExe = (Get-Process -Id $PID).Path
$BatchRoot = Join-Path $OutputRoot "_batch"
$LogRoot = Join-Path $BatchRoot "logs"

New-Item -ItemType Directory -Force -Path $BatchRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

if (-not (Test-Path -LiteralPath $Runner)) {
    throw ("Runner not found: " + $Runner)
}

function New-StatusRow {
    param(
        [string]$ParticipantId,
        [string]$RunStage,
        [int]$ExitCode,
        [string]$Message,
        [string]$StdOut,
        [string]$StdErr
    )

    return [pscustomobject]@{
        ParticipantId = $ParticipantId
        Stage = $RunStage
        ExitCode = $ExitCode
        Message = $Message
        StdOut = $StdOut
        StdErr = $StdErr
    }
}

function Export-StatusRows {
    param(
        [array]$Rows,
        [string]$Path
    )

    if ($Rows.Count -gt 0) {
        $Rows | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding UTF8
    }
    else {
        "ParticipantId,Stage,ExitCode,Message,StdOut,StdErr" | Set-Content -LiteralPath $Path -Encoding UTF8
    }
}

function Invoke-VisualQueue {
    param(
        [array]$Jobs,
        [int]$Limit,
        [string]$RunStage
    )

    $Pending = @($Jobs)
    $Running = @()
    $Completed = @()
    $Total = $Pending.Count

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor DarkCyan
    Write-Host (" DG3 DYNAMIC-AOI PIPELINE - " + $RunStage) -ForegroundColor Cyan
    Write-Host (" Participants: " + [string]$Total + " | Parallel: " + [string]$Limit)
    Write-Host "============================================================" -ForegroundColor DarkCyan

    while (($Pending.Count -gt 0) -or ($Running.Count -gt 0)) {
        while (($Pending.Count -gt 0) -and ($Running.Count -lt $Limit)) {
            $Job = $Pending[0]

            if ($Pending.Count -eq 1) {
                $Pending = @()
            }
            else {
                $Pending = @($Pending[1..($Pending.Count - 1)])
            }

            $StartParameters = @{
                FilePath = $PowerShellExe
                ArgumentList = $Job.ArgumentList
                RedirectStandardOutput = $Job.StdOut
                RedirectStandardError = $Job.StdErr
                WindowStyle = "Hidden"
                PassThru = $true
            }

            $Process = Start-Process @StartParameters

            $Running += [pscustomobject]@{
                ParticipantId = $Job.ParticipantId
                Process = $Process
                StartTime = Get-Date
                StdOut = $Job.StdOut
                StdErr = $Job.StdErr
                ExpectedOutput = $Job.ExpectedOutput
            }

            Write-Host ("[START] " + $Job.ParticipantId) -ForegroundColor Yellow
        }

        $Finished = @($Running | Where-Object { $_.Process.HasExited })

        foreach ($Item in $Finished) {
            $Item.Process.WaitForExit()
            $ExitCode = [int]$Item.Process.ExitCode
            $Message = "Completed"

            if ($ExitCode -eq 0) {
                if (-not [string]::IsNullOrWhiteSpace($Item.ExpectedOutput)) {
                    if (-not (Test-Path -LiteralPath $Item.ExpectedOutput)) {
                        $ExitCode = 2001
                        $Message = "Expected output not created"
                    }
                }
            }
            else {
                $Message = "Child process failed"
            }

            $Completed += New-StatusRow -ParticipantId $Item.ParticipantId -RunStage $RunStage -ExitCode $ExitCode -Message $Message -StdOut $Item.StdOut -StdErr $Item.StdErr

            if ($ExitCode -eq 0) {
                Write-Host ("[DONE ] " + $Item.ParticipantId) -ForegroundColor Green
            }
            else {
                Write-Host ("[FAIL ] " + $Item.ParticipantId + " ExitCode=" + [string]$ExitCode) -ForegroundColor Red
            }

            $Running = @($Running | Where-Object { $_.Process.Id -ne $Item.Process.Id })
        }

        $DoneCount = $Completed.Count
        $SuccessCount = @($Completed | Where-Object { $_.ExitCode -eq 0 }).Count
        $FailedCount = @($Completed | Where-Object { $_.ExitCode -ne 0 }).Count
        $Percent = 0

        if ($Total -gt 0) {
            $Percent = [int](100.0 * $DoneCount / $Total)
        }

        $RunningNames = @(
            $Running | ForEach-Object {
                $Elapsed = (Get-Date) - $_.StartTime
                $_.ParticipantId + " " + $Elapsed.ToString("hh\:mm\:ss")
            }
        )

        $StatusText = "Done " + [string]$DoneCount + "/" + [string]$Total + " | Success " + [string]$SuccessCount + " | Failed " + [string]$FailedCount + " | Running " + [string]$Running.Count + " | Queued " + [string]$Pending.Count

        Write-Progress -Activity ("DG3 dynamic-AOI pipeline - " + $RunStage) -Status $StatusText -PercentComplete $Percent
        Write-Host ("`r" + $StatusText + " | " + ($RunningNames -join " ; ") + "                              ") -NoNewline

        Start-Sleep -Seconds 2
    }

    Write-Progress -Activity ("DG3 dynamic-AOI pipeline - " + $RunStage) -Completed
    Write-Host ""
    Write-Host ($RunStage + " stage finished.") -ForegroundColor Cyan

    return $Completed
}

$ParticipantIds = @()
for ($Id = $StartId; $Id -le $EndId; $Id++) {
    $ParticipantIds += ("P{0:D2}" -f $Id)
}

$AllResults = @()

if (($Stage -eq "Metrics") -or ($Stage -eq "All")) {
    $MetricJobs = @()

    foreach ($ParticipantId in $ParticipantIds) {
        $Folder = Join-Path $ProjectRoot ("data_raw\" + $ParticipantId)
        $TimelineFile = Join-Path $Folder $TimelineName
        $StdOut = Join-Path $LogRoot ($ParticipantId + "_metrics.out.log")
        $StdErr = Join-Path $LogRoot ($ParticipantId + "_metrics.err.log")

        if (-not (Test-Path -LiteralPath $Folder)) {
            $AllResults += New-StatusRow -ParticipantId $ParticipantId -RunStage "Metrics" -ExitCode 1001 -Message "Participant folder not found" -StdOut "" -StdErr $Folder
            continue
        }

        $FolderFiles = @(Get-ChildItem -LiteralPath $Folder -File -ErrorAction SilentlyContinue)

        if ($FolderFiles.Count -eq 0) {
            $AllResults += New-StatusRow -ParticipantId $ParticipantId -RunStage "Metrics" -ExitCode 1002 -Message "Participant folder is empty" -StdOut "" -StdErr $Folder
            continue
        }

        if (-not (Test-Path -LiteralPath $TimelineFile)) {
            $AllResults += New-StatusRow -ParticipantId $ParticipantId -RunStage "Metrics" -ExitCode 1003 -Message "Timeline not found" -StdOut "" -StdErr $TimelineFile
            continue
        }

        $Arguments = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", ('"{0}"' -f $Runner),
            "-ProjectRoot", ('"{0}"' -f $ProjectRoot),
            "-OutputRoot", ('"{0}"' -f $OutputRoot),
            "-ParticipantId", $ParticipantId,
            "-TimelineFile", ('"{0}"' -f $TimelineFile),
            "-NoOverlay"
        )

        if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
            $Arguments += @("-PythonExe", ('"{0}"' -f $PythonExe))
        }

        if (-not [string]::IsNullOrWhiteSpace($ConfigFile)) {
            $Arguments += @("-ConfigFile", ('"{0}"' -f $ConfigFile))
        }

        if ($ReuseExistingMarkers) {
            $CandidateReuseDir = Join-Path $ReuseRoot $ParticipantId
            $Detection = Join-Path $CandidateReuseDir ($ParticipantId + "_windowed_marker_detections.csv")
            $FrameSummary = Join-Path $CandidateReuseDir ($ParticipantId + "_windowed_marker_frame_summary.csv")

            if ((Test-Path -LiteralPath $Detection) -and (Test-Path -LiteralPath $FrameSummary)) {
                $Arguments += @("-ReuseMarkerDir", ('"{0}"' -f $CandidateReuseDir))
            }
        }

        $ExpectedOutput = Join-Path (Join-Path $OutputRoot $ParticipantId) ($ParticipantId + "_toi_metrics.csv")

        $MetricJobs += [pscustomobject]@{
            ParticipantId = $ParticipantId
            ArgumentList = $Arguments
            StdOut = $StdOut
            StdErr = $StdErr
            ExpectedOutput = $ExpectedOutput
        }
    }

    $MetricResults = Invoke-VisualQueue -Jobs $MetricJobs -Limit $MaxParallel -RunStage "Metrics"
    $AllResults += $MetricResults

    $MetricStatus = @($AllResults | Where-Object { $_.Stage -eq "Metrics" })
    Export-StatusRows -Rows $MetricStatus -Path (Join-Path $BatchRoot "metrics_run_status.csv")
}

if (($Stage -eq "Overlay") -or ($Stage -eq "All")) {
    $OverlayPlan = @()
    $OverlayJobs = @()

    foreach ($ParticipantId in $ParticipantIds) {
        $ParticipantOutput = Join-Path $OutputRoot $ParticipantId
        $AvailabilityFile = Join-Path $ParticipantOutput ($ParticipantId + "_trial_aoi_availability.csv")
        $ToiFile = Join-Path $ParticipantOutput ($ParticipantId + "_timeline_toi.csv")

        if ((-not (Test-Path -LiteralPath $AvailabilityFile)) -or (-not (Test-Path -LiteralPath $ToiFile))) {
            $OverlayPlan += [pscustomobject]@{
                ParticipantId = $ParticipantId
                Status = "SKIP_NO_METRICS"
                LowAvailability = ""
                OverlayTrials = ""
                Reason = "Missing metrics outputs"
            }
            continue
        }

        $Availability = @(Import-Csv -LiteralPath $AvailabilityFile)
        $LowAvailability = $false

        foreach ($AoiName in @("Task", "Delivery", "RobotTotal")) {
            $Values = @(
                $Availability |
                Where-Object { $_.aoi_name -eq $AoiName } |
                ForEach-Object { [double]$_.availability_rate }
            )

            if ($Values.Count -eq 0) {
                $LowAvailability = $true
            }
            else {
                $MeanValue = [double](($Values | Measure-Object -Average).Average)
                if ($MeanValue -lt $LowAvailabilityThreshold) {
                    $LowAvailability = $true
                }
            }
        }

        $UsableTrials = @(
            Import-Csv -LiteralPath $ToiFile |
            Where-Object { $_.phase -eq "Trial_Overall" } |
            ForEach-Object { [int]$_.trial_index } |
            Sort-Object -Unique
        )

        if ($UsableTrials.Count -eq 0) {
            $OverlayPlan += [pscustomobject]@{
                ParticipantId = $ParticipantId
                Status = "SKIP_NO_USABLE_TRIAL"
                LowAvailability = $LowAvailability
                OverlayTrials = ""
                Reason = "No usable trials"
            }
            continue
        }

        if ($LowAvailability) {
            $OverlayTrials = $UsableTrials -join ","
            $Reason = "All usable trials because availability is low"
        }
        else {
            $ParticipantNumber = [int]$ParticipantId.Substring(1)
            $SeedValue = $RandomSeed + $ParticipantNumber
            $Random = New-Object -TypeName System.Random -ArgumentList $SeedValue
            $SelectedIndex = $Random.Next(0, $UsableTrials.Count)
            $OverlayTrials = [string]$UsableTrials[$SelectedIndex]
            $Reason = "One deterministic random usable trial"
        }

        $OverlayPlan += [pscustomobject]@{
            ParticipantId = $ParticipantId
            Status = "READY"
            LowAvailability = $LowAvailability
            OverlayTrials = $OverlayTrials
            Reason = $Reason
        }

        $Folder = Join-Path $ProjectRoot ("data_raw\" + $ParticipantId)
        $TimelineFile = Join-Path $Folder $TimelineName
        $StdOut = Join-Path $LogRoot ($ParticipantId + "_overlay.out.log")
        $StdErr = Join-Path $LogRoot ($ParticipantId + "_overlay.err.log")

        $Arguments = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", ('"{0}"' -f $Runner),
            "-ProjectRoot", ('"{0}"' -f $ProjectRoot),
            "-OutputRoot", ('"{0}"' -f $OutputRoot),
            "-ParticipantId", $ParticipantId,
            "-TimelineFile", ('"{0}"' -f $TimelineFile),
            "-ReuseMarkerDir", ('"{0}"' -f $ParticipantOutput),
            "-OverlayTrialIndices", $OverlayTrials
        )

        if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
            $Arguments += @("-PythonExe", ('"{0}"' -f $PythonExe))
        }

        if (-not [string]::IsNullOrWhiteSpace($ConfigFile)) {
            $Arguments += @("-ConfigFile", ('"{0}"' -f $ConfigFile))
        }

        $OverlayJobs += [pscustomobject]@{
            ParticipantId = $ParticipantId
            ArgumentList = $Arguments
            StdOut = $StdOut
            StdErr = $StdErr
            ExpectedOutput = ""
        }
    }

    $OverlayPlan | Export-Csv -LiteralPath (Join-Path $BatchRoot "overlay_plan.csv") -NoTypeInformation -Encoding UTF8
    $OverlayResults = Invoke-VisualQueue -Jobs $OverlayJobs -Limit $MaxParallel -RunStage "Overlay"
    $AllResults += $OverlayResults
    Export-StatusRows -Rows $OverlayResults -Path (Join-Path $BatchRoot "overlay_run_status.csv")
}

Export-StatusRows -Rows $AllResults -Path (Join-Path $BatchRoot "all_run_status.csv")

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host " BATCH COMPLETE" -ForegroundColor Green
Write-Host (" Output: " + $OutputRoot)
Write-Host (" QA: " + $BatchRoot)
Write-Host "============================================================" -ForegroundColor DarkCyan
