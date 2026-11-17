@echo off
REM Joins the 4 parts back into circuit_identification.zip (353 MB).
REM Double-click this file after downloading all 4 parts into the same folder.
cd /d "%~dp0"
for %%f in (circuit_identification.zip.001 circuit_identification.zip.002 circuit_identification.zip.003 circuit_identification.zip.004) do (
  if not exist %%f (
    echo Missing %%f - download all 4 parts into this folder first.
    pause
    exit /b 1
  )
)
copy /b circuit_identification.zip.001+circuit_identification.zip.002+circuit_identification.zip.003+circuit_identification.zip.004 circuit_identification.zip >nul
echo.
echo Created circuit_identification.zip. Checking it...
powershell -NoProfile -Command "$h=(Get-FileHash circuit_identification.zip).Hash; if($h -eq '4EF5E88F3D91B18A6010F15FD850A0F899510B6D426D88C0B23AFFBAE8D39D11'){Write-Host 'OK - file is identical to the original.'}else{Write-Host 'WARNING - checksum mismatch, re-download the parts.' $h}"
echo.
pause
