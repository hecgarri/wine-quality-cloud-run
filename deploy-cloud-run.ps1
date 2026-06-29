[CmdletBinding()]
param(
    [string]$ProjectId = "",
    [string]$Region = "us-central1",
    [string]$ServiceName = "wine-quality-predictor",
    [string]$ArtifactRepository = "cloud-run-images",
    [string]$ImageName = "wine-quality-app",
    [string]$ModelBucket = "",
    [string]$ModelPath = "app/model.keras",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$gcloud = Get-Command gcloud.cmd -ErrorAction SilentlyContinue
if ($null -eq $gcloud) {
    $gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
}
if ($null -eq $gcloud) {
    throw "Google Cloud CLI no está disponible en PATH."
}

function Invoke-Gcloud {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & $gcloud.Source @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud falló: gcloud $($Arguments -join ' ')"
    }
}

function Get-GcloudValue {
    param([Parameter(Mandatory)][string[]]$Arguments)
    $output = & $gcloud.Source @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud falló: gcloud $($Arguments -join ' ')"
    }
    return ($output | Select-Object -Last 1).Trim()
}

function Test-GcloudResource {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & $gcloud.Source @Arguments *> $null
    return $LASTEXITCODE -eq 0
}

if ([string]::IsNullOrWhiteSpace($ProjectId)) {
    $ProjectId = Get-GcloudValue @("config", "get-value", "project")
}
if ([string]::IsNullOrWhiteSpace($ProjectId) -or $ProjectId -eq "(unset)") {
    throw "No hay un proyecto configurado. Usa -ProjectId o gcloud config set project."
}

$repositoryRoot = $PSScriptRoot
$resolvedModelPath = if ([IO.Path]::IsPathRooted($ModelPath)) {
    [IO.Path]::GetFullPath($ModelPath)
} else {
    [IO.Path]::GetFullPath((Join-Path $repositoryRoot $ModelPath))
}
if (-not (Test-Path -LiteralPath $resolvedModelPath -PathType Leaf)) {
    throw "No existe $resolvedModelPath. Ejecuta parte0/train.py antes de desplegar."
}

if ([string]::IsNullOrWhiteSpace($ModelBucket)) {
    $ModelBucket = "$ProjectId-ml-models"
}

$modelHash = (Get-FileHash -LiteralPath $resolvedModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
$shortHash = $modelHash.Substring(0, 12)
$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
$modelUri = "gs://$ModelBucket/wine-quality/$modelHash/model.keras"
$imageTag = "$timestamp-$shortHash"
$imageUri = "$Region-docker.pkg.dev/$ProjectId/$ArtifactRepository/$ImageName`:$imageTag"

Write-Host "Proyecto: $ProjectId"
Write-Host "Modelo: $modelUri"
Write-Host "Imagen: $imageUri"
Write-Host "Servicio: $ServiceName ($Region)"

if ($DryRun) {
    Write-Host "DryRun: no se crearon ni modificaron recursos."
    Write-Host "Se habilitarían APIs, se publicaría el modelo, se construiría la imagen y se desplegaría Cloud Run."
    return
}

$account = Get-GcloudValue @("auth", "list", "--filter=status:ACTIVE", "--format=value(account)")
if ([string]::IsNullOrWhiteSpace($account)) {
    throw "No hay una cuenta activa en Google Cloud CLI."
}

Invoke-Gcloud @(
    "services", "enable",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "--project=$ProjectId",
    "--quiet"
)

$bucketUri = "gs://$ModelBucket"
if (-not (Test-GcloudResource @("storage", "buckets", "describe", $bucketUri, "--project=$ProjectId"))) {
    Invoke-Gcloud @(
        "storage", "buckets", "create", $bucketUri,
        "--project=$ProjectId",
        "--location=$Region",
        "--uniform-bucket-level-access"
    )
}

Invoke-Gcloud @("storage", "cp", $resolvedModelPath, $modelUri, "--no-clobber")

if (-not (Test-GcloudResource @(
    "artifacts", "repositories", "describe", $ArtifactRepository,
    "--project=$ProjectId", "--location=$Region"
))) {
    Invoke-Gcloud @(
        "artifacts", "repositories", "create", $ArtifactRepository,
        "--project=$ProjectId",
        "--location=$Region",
        "--repository-format=docker",
        "--description=Imágenes de Cloud Run"
    )
}

$buildContext = Join-Path ([IO.Path]::GetTempPath()) "wine-quality-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $buildContext | Out-Null

try {
    foreach ($file in @("Dockerfile", "app.py", "pyproject.toml", "poetry.lock")) {
        Copy-Item -LiteralPath (Join-Path $repositoryRoot "app/$file") -Destination $buildContext
    }
    Invoke-Gcloud @("storage", "cp", $modelUri, (Join-Path $buildContext "model.keras"))
    Invoke-Gcloud @(
        "builds", "submit", $buildContext,
        "--project=$ProjectId",
        "--tag=$imageUri",
        "--quiet"
    )
    Invoke-Gcloud @(
        "run", "deploy", $ServiceName,
        "--project=$ProjectId",
        "--region=$Region",
        "--platform=managed",
        "--image=$imageUri",
        "--allow-unauthenticated",
        "--port=8501",
        "--min=0",
        "--max-instances=1",
        "--cpu=1",
        "--memory=1Gi",
        "--concurrency=10",
        "--timeout=300",
        "--cpu-throttling",
        "--quiet"
    )
} finally {
    if (Test-Path -LiteralPath $buildContext) {
        Remove-Item -LiteralPath $buildContext -Recurse -Force
    }
}

$serviceUrl = Get-GcloudValue @(
    "run", "services", "describe", $ServiceName,
    "--project=$ProjectId",
    "--region=$Region",
    "--format=value(status.url)"
)
Write-Host "Despliegue completado: $serviceUrl"
Write-Host "Modelo desplegado: $modelUri"
Write-Host "Imagen desplegada: $imageUri"
