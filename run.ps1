Write-Host "Compiling VectorDB Java engine..." -ForegroundColor Cyan
javac (Get-ChildItem -Path . -Filter *.java | ForEach-Object { $_.FullName })
if ($LASTEXITCODE -ne 0) {
    Write-Error "Compilation failed!"
    exit $LASTEXITCODE
}
Write-Host "Starting VectorDB server on http://localhost:8080..." -ForegroundColor Green
java Main
