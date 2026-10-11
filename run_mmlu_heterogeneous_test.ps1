[CmdletBinding()]
param(
    [ValidateRange(1, 2000)]
    [int]$DataLimit = 200,

    [string]$PolicyCheckpoint = "puppeteer\checkpoint\MMLU-Pro_train\policy_net_20261011_075840.pt",

    [switch]$SkipRewardSmoke,

    [switch]$LiveProviders,

    [switch]$PreflightOnly,

    [switch]$Resume,

    [ValidateRange(1, 2000)]
    [int]$CheckpointEvery = 1
)

$ErrorActionPreference = "Stop"

$mutexName = "Local\RACO_MAS_MMLU_Pro_Heterogeneous_Run"
$mutexCreated = $false
$mutex = [System.Threading.Mutex]::new($true, $mutexName, [ref]$mutexCreated)
if (-not $mutexCreated) {
    $mutex.Dispose()
    throw "Another MMLU-Pro heterogeneous train/test run is already running."
}

try {
    $repoRoot = $PSScriptRoot
    $python = Join-Path $repoRoot ".venv311\Scripts\python.exe"
    $puppeteerRoot = Join-Path $repoRoot "puppeteer"
    $pool = "personas/mmlu_pro_heterogeneous_pool.jsonl"
    $resultPath = Join-Path $puppeteerRoot "results\MMLU-Pro_test\MMLU-Pro_test.jsonl"
    $testCheckpoint = Join-Path $puppeteerRoot "checkpoint\MMLU-Pro_test\resume_latest.pt"
    $trainResult = Join-Path $puppeteerRoot "results\MMLU-Pro_train\MMLU-Pro_train.jsonl"

    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Python environment not found: $python"
    }

    if ([IO.Path]::IsPathRooted($PolicyCheckpoint)) {
        $trainedCheckpoint = $PolicyCheckpoint
    }
    else {
        $trainedCheckpoint = Join-Path $repoRoot $PolicyCheckpoint
    }
    if (-not (Test-Path -LiteralPath $trainedCheckpoint -PathType Leaf)) {
        throw "Trained policy checkpoint not found: $trainedCheckpoint"
    }
    $trainedCheckpoint = (Resolve-Path -LiteralPath $trainedCheckpoint).Path

    $pythonVersion = & $python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
    if ($LASTEXITCODE -ne 0 -or -not $pythonVersion.StartsWith("3.11.")) {
        throw "MMLU-Pro test requires Python 3.11; found $pythonVersion"
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
    Write-Host "Evaluation policy: $trainedCheckpoint"

    if (-not (Test-Path -LiteralPath $trainResult -PathType Leaf)) {
        throw "Verified 400-row training result not found: $trainResult"
    }
    $trainRecords = @(
        Get-Content -LiteralPath $trainResult |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_ | ConvertFrom-Json }
    )
    $trainUniqueIds = @($trainRecords.id | Sort-Object -Unique).Count
    $trainInvalid = @(
        $trainRecords | Where-Object {
            ([string]$_.pred).Trim() -notmatch '^[A-J]$'
        }
    )
    if (
        $trainRecords.Count -ne 400 -or
        $trainUniqueIds -ne 400 -or
        $trainInvalid.Count -ne 0
    ) {
        throw "Training artifact verification failed; expected 400 valid unique rows"
    }
    Write-Host "Training artifact: verified 400 rows"

    $trainedCheckpointHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $trainedCheckpoint).Hash
    Write-Host "Policy SHA256: $trainedCheckpointHash"

    Write-Host "Running local regression checks..."
    Push-Location $repoRoot
    try {
        & $python -X utf8 -m unittest `
            tests.test_reasoning_resilience `
            tests.test_dataset_splits `
            tests.test_persona_tools
        if ($LASTEXITCODE -ne 0) {
            throw "Regression checks failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }

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

        Write-Host "Checking the seed-42 final split..."
        & $python -X utf8 -c "import sys; from tasks.mmlu_pro import load_dataset; n=int(sys.argv[1]); rows=load_dataset('test', data_limit=n, seed=42); assert len(rows)==n; assert rows.question_id.nunique()==n; print(f'Final split prefix: {n} unique rows')" $DataLimit
        if ($LASTEXITCODE -ne 0) {
            throw "Test split verification failed with exit code $LASTEXITCODE"
        }

        if ($PreflightOnly) {
            Write-Host "Preflight completed successfully; test was not started."
            return
        }

        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $dataStart = 0
        $runLimit = $DataLimit
        $runCheckpoint = $trainedCheckpoint
        $resumeArguments = @()

        if ($Resume) {
            if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
                throw "Cannot resume because test result is missing: $resultPath"
            }
            if (-not (Test-Path -LiteralPath $testCheckpoint -PathType Leaf)) {
                throw "Cannot resume because test checkpoint is missing: $testCheckpoint"
            }
            $dataStart = @(
                Get-Content -LiteralPath $resultPath |
                    Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
            ).Count
            if ($dataStart -ge $DataLimit) {
                throw "Nothing to resume: test already has $dataStart row(s), target is $DataLimit."
            }
            $runLimit = $DataLimit - $dataStart
            $runCheckpoint = $testCheckpoint
            $resumeArguments = @("--data_start", $dataStart)
            Write-Host "Resuming test from $dataStart/$DataLimit completed sample(s)."
        }
        else {
            if (Test-Path -LiteralPath $resultPath -PathType Leaf) {
                $resultBackupDirectory = Join-Path $puppeteerRoot "results\backups"
                New-Item -ItemType Directory -Path $resultBackupDirectory -Force | Out-Null
                $resultBackup = Join-Path $resultBackupDirectory "MMLU-Pro_test_before_heterogeneous_$timestamp.jsonl"
                Copy-Item -LiteralPath $resultPath -Destination $resultBackup
                Write-Host "Previous test result backed up to: $resultBackup"
            }
            if (Test-Path -LiteralPath $testCheckpoint -PathType Leaf) {
                $checkpointBackupDirectory = Join-Path $puppeteerRoot "checkpoint\backups"
                New-Item -ItemType Directory -Path $checkpointBackupDirectory -Force | Out-Null
                $checkpointBackup = Join-Path $checkpointBackupDirectory "MMLU-Pro_test_resume_$timestamp.pt"
                Move-Item -LiteralPath $testCheckpoint -Destination $checkpointBackup
                Write-Host "Previous test resume checkpoint moved to: $checkpointBackup"
            }
        }

        $runTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $consoleLog = Join-Path $puppeteerRoot "mmlu_pro_heterogeneous_test_seed42_$runTimestamp.log"
        Write-Host "Starting frozen-policy MMLU-Pro test: $runLimit sample(s) this invocation"
        Write-Host "Console log: $consoleLog"
        Write-Host "Result: $resultPath"
        Write-Host "Test resume checkpoint: $testCheckpoint"

        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $benchmarkArguments = @(
                "-W", "ignore::RuntimeWarning:pydub.utils",
                "-X", "utf8",
                "main.py", "MMLU-Pro", "test",
                "--data_limit", $runLimit,
                "--checkpoint_every", $CheckpointEvery,
                "--checkpoint", $runCheckpoint,
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
            throw "MMLU-Pro test failed with exit code $benchmarkExitCode"
        }

        $completedRecords = @(
            Get-Content -LiteralPath $resultPath |
                Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
                ForEach-Object { $_ | ConvertFrom-Json }
        )
        $invalidRecords = @(
            $completedRecords | Where-Object {
                ([string]$_.pred).Trim() -notmatch '^[A-J]$'
            }
        )
        $uniqueIds = @($completedRecords.id | Sort-Object -Unique).Count
        if ($completedRecords.Count -ne $DataLimit) {
            throw "Result verification failed: expected $DataLimit rows, found $($completedRecords.Count)"
        }
        if ($uniqueIds -ne $DataLimit) {
            throw "Result verification failed: expected $DataLimit unique IDs, found $uniqueIds"
        }
        if ($invalidRecords.Count -gt 0) {
            throw "Result verification failed: found $($invalidRecords.Count) invalid prediction(s)"
        }

        $trainIds = @($trainRecords.id | ForEach-Object { [string]$_ })
        $overlap = @(
            $completedRecords | Where-Object { $trainIds -contains [string]$_.id }
        )
        if ($overlap.Count -gt 0) {
            throw "Split verification failed: test contains $($overlap.Count) training ID(s)"
        }

        $sourceHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $trainedCheckpoint).Hash
        if ($sourceHashAfter -ne $trainedCheckpointHash) {
            throw "The source training checkpoint changed during evaluation"
        }

        $correctCount = @($completedRecords | Where-Object { $_.correct -eq $true }).Count
        $accuracy = [Math]::Round($correctCount / $DataLimit, 6)
        $summaryPath = Join-Path (Split-Path $resultPath -Parent) "summary_seed42_$DataLimit.json"
        $summary = [ordered]@{
            task = "MMLU-Pro"
            mode = "test"
            seed = 42
            data_limit = $DataLimit
            correct = $correctCount
            accuracy = $accuracy
            unique_ids = $uniqueIds
            invalid_predictions = $invalidRecords.Count
            train_test_id_overlap = $overlap.Count
            policy_checkpoint = $trainedCheckpoint
            policy_sha256 = $trainedCheckpointHash
            result = $resultPath
            console_log = $consoleLog
            completed_at = (Get-Date).ToString("o")
        }
        $summary | ConvertTo-Json | Set-Content -LiteralPath $summaryPath -Encoding UTF8

        Write-Host "Test completed successfully with a frozen policy."
        Write-Host "Verified: $DataLimit rows, $uniqueIds unique IDs, 0 invalid predictions, 0 train/test ID overlap."
        Write-Host "Accuracy: $correctCount/$DataLimit = $accuracy"
        Write-Host "Result: $resultPath"
        Write-Host "Summary: $summaryPath"
        Write-Host "Console log: $consoleLog"
        Write-Host "Resume checkpoint: $testCheckpoint"
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -ne $mutex) {
        $mutex.ReleaseMutex()
        $mutex.Dispose()
    }
}
