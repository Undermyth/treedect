# treedect 打包脚本
# 用法: .\package.ps1

param(
    [switch]$Release = $true,
    [switch]$Clean = $false,
    [switch]$SkipBuild = $false
)

$ErrorActionPreference = "Stop"

# 颜色输出函数
function Write-Color($Text, $Color) {
    Write-Host $Text -ForegroundColor $Color
}

Write-Color "=== treedect 打包脚本 ===" "Cyan"

# 1. 清理旧的构建（可选）
if ($Clean) {
    Write-Color "Cleaning old builds..." "Yellow"
    if (Test-Path "target\release") {
        Remove-Item -Recurse -Force "target\release" -ErrorAction SilentlyContinue
    }
    if (Test-Path "dist") {
        Remove-Item -Recurse -Force "dist" -ErrorAction SilentlyContinue
    }
}

# 2. 构建 Release 版本
if (-not $SkipBuild) {
    Write-Color "Building Release version..." "Yellow"

    # 检查 Python 环境
    if (-not (Test-Path "pyenv\install\python.exe")) {
        Write-Color "Error: Python environment not found at pyenv\install" "Red"
        exit 1
    }

    # 构建
    cargo build --release

    if ($LASTEXITCODE -ne 0) {
        Write-Color "Build failed!" "Red"
        exit 1
    }

    Write-Color "Build successful!" "Green"
} else {
    Write-Color "Skipping build (using existing binary)" "Yellow"
}

# 3. 创建输出目录
$outputDir = "dist\treedect"
Write-Color "Creating output directory: $outputDir" "Yellow"

if (Test-Path $outputDir) {
    Remove-Item -Recurse -Force $outputDir
}
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# 4. 复制二进制文件
$binaryPath = "target\release\treedect.exe"
if (-not (Test-Path $binaryPath)) {
    Write-Color "Error: Binary not found at $binaryPath" "Red"
    exit 1
}

Write-Color "Copying binary..." "Yellow"
Copy-Item $binaryPath $outputDir

# 5. 复制核心 DLL（这些应该在 build.rs 中已复制到 target\release）
$coreDlls = @(
    "python311.dll",
    "python3.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll"
)

Write-Color "Copying Python DLLs..." "Yellow"
foreach ($dll in $coreDlls) {
    $dllPath = "target\release\$dll"
    if (Test-Path $dllPath) {
        Copy-Item $dllPath $outputDir
        Write-Color "  Copied: $dll" "Gray"
    } else {
        Write-Color "  Warning: $dll not found in target\release" "Yellow"
        # 尝试从 pyenv 复制
        $pyenvDll = "pyenv\install\$dll"
        if (Test-Path $pyenvDll) {
            Copy-Item $pyenvDll $outputDir
            Write-Color "  Copied from pyenv: $dll" "Gray"
        }
    }
}

# 5.5 复制 DirectML.dll（ONNX Runtime 依赖）
Write-Color "Copying DirectML.dll..." "Yellow"
$directmlPath = "target\release\DirectML.dll"
if (Test-Path $directmlPath) {
    Copy-Item $directmlPath $outputDir
    Write-Color "  Copied: DirectML.dll" "Green"
} else {
    Write-Color "  Warning: DirectML.dll not found in target\release" "Yellow"
}

# 6. 复制 Python 环境（由 build.rs 生成）
$pythonEnvSource = "target\release\python"
$pythonEnvDest = "$outputDir\python"

if (Test-Path $pythonEnvSource) {
    Write-Color "Copying Python environment..." "Yellow"
    Copy-Item -Recurse -Force $pythonEnvSource $pythonEnvDest
    Write-Color "  Python environment copied" "Green"
    
    # 验证关键目录是否存在
    Write-Color "Verifying Python environment..." "Yellow"
    $requiredPaths = @(
        "Lib\encodings",
        "Lib\site-packages",
        "DLLs"
    )
    
    $allOk = $true
    foreach ($reqPath in $requiredPaths) {
        $fullPath = Join-Path $pythonEnvDest $reqPath
        if (Test-Path $fullPath) {
            $itemCount = (Get-ChildItem -Path $fullPath -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
            Write-Color "  [OK] $reqPath ($itemCount items)" "Green"
        } else {
            Write-Color "  [MISSING] $reqPath" "Red"
            $allOk = $false
        }
    }
    
    if (-not $allOk) {
        Write-Color "ERROR: Some required Python directories are missing!" "Red"
        Write-Color "The application may not work correctly." "Red"
    }
} else {
    Write-Color "ERROR: Python environment not found at $pythonEnvSource" "Red"
    Write-Color "The build.rs script should have created this during compilation." "Red"
    Write-Color "Please check the build output for errors." "Red"
    exit 1
}

# 7. 清理编译缓存文件（保留所有功能模块，包括测试相关）
Write-Color "Cleaning up cache files..." "Yellow"

# 删除 __pycache__
Get-ChildItem -Path $pythonEnvDest -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 删除 .pyc 和 .pyo 文件
Get-ChildItem -Path $pythonEnvDest -Recurse -Include "*.pyc", "*.pyo" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

# 8. 显示最终大小
Write-Color "" "White"
Write-Color "=== 打包完成 ===" "Green"

$outputSize = (Get-ChildItem -Path $outputDir -Recurse | Measure-Object -Property Length -Sum).Sum
$outputSizeMB = [math]::Round($outputSize / 1MB, 2)

Write-Color "输出目录: $outputDir" "Cyan"
Write-Color "总大小: $outputSizeMB MB" "Cyan"
Write-Color "" "White"

# 9. 显示目录结构
Write-Color "目录结构:" "Cyan"
Get-ChildItem -Path $outputDir | ForEach-Object {
    if ($_.PSIsContainer) {
        $size = (Get-ChildItem -Path $_.FullName -Recurse | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 2)
        Write-Color "  [DIR]  $($_.Name) ($sizeMB MB)" "Gray"
    } else {
        $sizeKB = [math]::Round($_.Length / 1KB, 2)
        Write-Color "  [FILE] $($_.Name) ($sizeKB KB)" "Gray"
    }
}

# 10. 可选：创建压缩包
$createZip = Read-Host "是否创建 ZIP 压缩包? (y/n)"
if ($createZip -eq "y" -or $createZip -eq "Y") {
    $zipPath = "dist\treedect.zip"
    Write-Color "Creating ZIP archive: $zipPath" "Yellow"

    if (Test-Path $zipPath) {
        Remove-Item $zipPath -Force
    }

    Compress-Archive -Path "$outputDir\*" -DestinationPath $zipPath -Force

    $zipSize = (Get-Item $zipPath).Length
    $zipSizeMB = [math]::Round($zipSize / 1MB, 2)
    Write-Color "ZIP created: $zipPath ($zipSizeMB MB)" "Green"
}

Write-Color "" "White"
Write-Color "=== 打包流程完成 ===" "Green"
Write-Color "可执行文件位于: $outputDir\treedect.exe" "Cyan"
Write-Color "直接运行即可，无需额外安装 Python" "Cyan"
