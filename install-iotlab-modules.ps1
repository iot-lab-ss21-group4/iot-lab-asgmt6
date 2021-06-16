Set-Location common
Get-ChildItem -Directory | ForEach-Object {
    Set-Location $_.BaseName
    if (Test-Path "setup.py" -PathType leaf)
    {
        pip install .
    }
    Set-Location ..
}
Set-Location ..
