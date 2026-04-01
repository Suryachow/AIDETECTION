# JustDone AI - Automated GitHub Push Script (Repo-Relative Version)
$RepoUrl = "https://github.com/Suryachow/AIDETECTION.git"
$RootPath = "D:\jd-2"
$TargetDir = Get-Location # Current folder (AIDETECTION)
$BundleDir = "$RootPath\MIGRATION_BUNDLE"
$SourceDir = "$RootPath\ScribePro\JD\JD"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   JustDone AI - GitHub Sync Agent" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Update from GitHub
Write-Host "`n[1/4] Pulling latest changes..." -ForegroundColor Yellow
git pull origin main

# 2. Copy Bundled Files (Fixing AI Detection)
Write-Host "`n[2/4] Syncing AI Detection & Humanization bundle..." -ForegroundColor Yellow
xcopy /E /I /Y "$BundleDir\*" "$TargetDir\"

# 3. Copy Humanizer Engine
Write-Host "`n[3/4] Porting Khizer Neural Humanizer Engine..." -ForegroundColor Yellow
xcopy /E /I /Y "$SourceDir\backend\khizer_humanizer" "$TargetDir\backend\khizer_humanizer\"

# 4. Commit and Push
Write-Host "`n[4/4] Pushing AI Detection fixes to GitHub..." -ForegroundColor Yellow
git add .
git commit -m "Merge AI Detection logic, Lexical Agent, and Writing endpoints"
git push origin main

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "   Push Complete! AI Detection is now FULLY updated." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
