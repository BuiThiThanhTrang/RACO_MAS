[CmdletBinding()]
param(
    [ValidateRange(1, 400)]
    [int]$DataLimit = 400,

    [switch]$SkipRewardSmoke,

    [switch]$LiveProviders,

    [switch]$PreflightOnly,

    [switch]$Resume,

    [ValidateRange(1, 400)]
    [int]$CheckpointEvery = 1
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$python = Join-Path $repoRoot ".venv311\Scripts\python.exe"
$puppeteerRoot = Join-Path $repoRoot "puppeteer"
$pool = "personas/mmlu_pro_heterogeneous_pool.jsonl"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

$pythonVersion = & $python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($LASTEXITCODE -ne 0 -or -not $pythonVersion.StartsWith("3.11.")) {
    throw "Baseline requires Python 3.11; found $pythonVersion"
}

$requiredVariables = @(
    "HF_TOKEN",
    "OPENROUTER_API_KEY",
    "REWARD_MODEL_URL",
    "MODAL_KEY",
    "MODAL_SECRET"
)
$missingVariables = @(
    $requiredVariables | Where-Object {
        [string]::IsNullOrWhiteSpace(
            [Environment]::GetEnvironmentVariable($_, "Process")
        )
    }
)
if ($missingVariables.Count -gt 0) {
    throw "Missing environment variables: $($missingVariables -join ', ')"
}
if (-not $env:MODAL_KEY.StartsWith("wk-")) {
    throw "MODAL_KEY must be a Proxy Token ID beginning with wk-"
}
if (-not $env:MODAL_SECRET.StartsWith("ws-")) {
    throw "MODAL_SECRET must be a Proxy Token secret beginning with ws-"
}
if (-not $env:REWARD_MODEL_URL.StartsWith("https://")) {
    throw "REWARD_MODEL_URL must be an HTTPS endpoint"
}

Write-Host "Python: $pythonVersion"
Write-Host "Environment variables: present"

if (-not $SkipRewardSmoke) {
    Write-Host "Checking the Modal reward endpoint..."
    Push-Location $repoRoot
    try {
        & $python -X utf8 -m deploy.reward_model.smoke_test
        if ($LASTEXITCODE -ne 0) {
            throw "Reward model smoke test failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

Push-Location $puppeteerRoot
try {
    Write-Host "Checking provider routing for $pool..."
    $providerArguments = @(
        "-X", "utf8", "provider_smoke_test.py",
        "--personas", $pool
    )
    if ($LiveProviders) {
        $providerArguments += "--live"
    }
    & $python @providerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Provider smoke test failed with exit code $LASTEXITCODE"
    }
    if ($PreflightOnly) {
        Write-Host "Preflight completed successfully; baseline was not started."
        return
    }

    $resultPath = Join-Path $puppeteerRoot "results\MMLU-Pro_train\MMLU-Pro_train.jsonl"
    $checkpointPath = Join-Path $puppeteerRoot "checkpoint\MMLU-Pro_train\resume_latest.pt"
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $dataStart = 0
    $runLimit = $DataLimit
    $resumeArguments = @()

    if ($Resume) {
        if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
            throw "Cannot resume because result file is missing: $resultPath"
        }
        if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
            throw "Cannot resume because checkpoint is missing: $checkpointPath"
        }
        $dataStart = @(
            Get-Content -LiteralPath $resultPath |
                Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        ).Count
        if ($dataStart -ge $DataLimit) {
            throw "Nothing to resume: result already has $dataStart row(s), target is $DataLimit."
        }
        $runLimit = $DataLimit - $dataStart
        $resumeArguments = @(
            "--data_start", $dataStart,
            "--checkpoint", $checkpointPath
        )
        Write-Host "Resuming from $dataStart/$DataLimit completed sample(s)."
    }
    elseif (Test-Path -LiteralPath $resultPath -PathType Leaf) {
        $backupDirectory = Join-Path $puppeteerRoot "results\backups"
        New-Item -ItemType Directory -Path $backupDirectory -Force | Out-Null
        $backupPath = Join-Path $backupDirectory "MMLU-Pro_train_before_heterogeneous_$timestamp.jsonl"
        Copy-Item -LiteralPath $resultPath -Destination $backupPath
        Write-Host "Previous result backed up to: $backupPath"
    }

    if (-not $Resume -and (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        $checkpointBackupDirectory = Join-Path $puppeteerRoot "checkpoint\backups"
        New-Item -ItemType Directory -Path $checkpointBackupDirectory -Force | Out-Null
        $checkpointBackup = Join-Path $checkpointBackupDirectory "MMLU-Pro_train_resume_$timestamp.pt"
        Move-Item -LiteralPath $checkpointPath -Destination $checkpointBackup
        Write-Host "Previous resume checkpoint moved to: $checkpointBackup"
    }

    $runTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $consoleLog = Join-Path $puppeteerRoot "mmlu_pro_heterogeneous_seed42_$runTimestamp.log"
    if ($Resume) {
        Write-Host "Continuing MMLU-Pro train with $runLimit remaining sample(s)..."
    }
    else {
        Write-Host "Starting MMLU-Pro train from scratch with $DataLimit sample(s)..."
    }
    Write-Host "Console log: $consoleLog"
    Write-Host "Resume checkpoint: $checkpointPath"

    # Windows PowerShell wraps any native stderr line (including harmless Python
    # warnings) as NativeCommandError. Let the process finish and judge success
    # using its exit code instead.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $benchmarkArguments = @(
            "-W", "ignore::RuntimeWarning:pydub.utils",
            "-X", "utf8",
            "main.py", "MMLU-Pro", "train",
            "--data_limit", $runLimit,
            "--checkpoint_every", $CheckpointEvery,
            "--seed", 42,
            "--personas", $pool
        ) + $resumeArguments
        & $python @benchmarkArguments 2>&1 | Tee-Object -FilePath $consoleLog
        $benchmarkExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($benchmarkExitCode -ne 0) {
        throw "Baseline failed with exit code $benchmarkExitCode"
    }

    Write-Host "Baseline completed successfully."
    Write-Host "Result: $resultPath"
    Write-Host "Resume checkpoint: $checkpointPath"
}
finally {
    Pop-Location
}
