# Load .env and start backend — run this instead of uvicorn directly
$envFile = Join-Path $PSScriptRoot ".env"
foreach ($line in Get-Content $envFile) {
    if ($line -match '^\s*#') { continue }       # skip comments
    if ($line -match '^\s*$') { continue }        # skip blank lines
    $parts = $line -split '=', 2
    if ($parts.Count -eq 2) {
        $key   = $parts[0].Trim()
        $value = $parts[1].Trim()
        [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
        Set-Item -Path "Env:$key" -Value $value
    }
}
Write-Host "Environment loaded from .env"
Write-Host "GROQ_API_KEY set: $($env:GROQ_API_KEY.Substring(0,10))..."
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
