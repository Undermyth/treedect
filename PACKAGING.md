# treedect 打包说明

## 文件说明

本次打包添加了以下文件：

1. **build.rs** - Rust 构建脚本，负责在编译时复制 Python 环境到输出目录
2. **package.ps1** - PowerShell 打包脚本，用于创建最终的发布包
3. **.cargo/config.toml** - Cargo 配置，设置 PyO3 使用的 Python 解释器路径
4. **修改了 src/models/cluster.rs** - 添加 Python 环境初始化代码
5. **修改了 Cargo.toml** - 添加 build.rs 和 build-dependencies

## 打包流程

### 方法一：使用 package.ps1（推荐）

最简单的打包方式，一键完成所有步骤：

```powershell
# 完整打包（构建 + 复制 + 清理）
.\package.ps1

# 跳过构建（使用已编译的二进制）
.\package.ps1 -SkipBuild

# 清理并重新构建
.\package.ps1 -Clean
```

### 方法二：手动打包

如果你需要更多控制，可以手动执行以下步骤：

#### 步骤 1：构建 Release 版本

```powershell
cargo build --release
```

构建过程中，`build.rs` 会自动：
- 复制 Python 核心 DLL 到 `target/release/`
- 复制精简后的 Python 环境到 `target/release/python/`

#### 步骤 2：创建发布目录

```powershell
# 创建输出目录
New-Item -ItemType Directory -Force -Path "dist\treedect"

# 复制二进制
copy "target\release\treedect.exe" "dist\treedect\"

# 复制核心 DLL
copy "target\release\python311.dll" "dist\treedect\"
copy "target\release\python3.dll" "dist\treedect\"
copy "target\release\vcruntime140.dll" "dist\treedect\"
copy "target\release\vcruntime140_1.dll" "dist\treedect\"

# 复制 Python 环境
xcopy /E /I "target\release\python" "dist\treedect\python"
```

#### 步骤 3：清理优化

```powershell
# 删除测试文件
Get-ChildItem -Path "dist\treedect\python" -Recurse -Filter "*test*" | Remove-Item -Recurse -Force

# 删除缓存
Get-ChildItem -Path "dist\treedect\python" -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path "dist\treedect\python" -Include "*.pyc", "*.pyo" -Recurse | Remove-Item -Force
```

#### 步骤 4：测试运行

```powershell
# 测试可执行文件
cd dist\treedect
.\treedect.exe
```

## 目录结构

打包后的目录结构：

```
dist/treedect/
├── treedect.exe              # Rust 可执行文件
├── python311.dll             # Python 核心 DLL
├── python3.dll               # Python 3 DLL
├── vcruntime140.dll          # MSVC 运行时
├── vcruntime140_1.dll        # MSVC 运行时
└── python/                   # Python 环境目录
    ├── DLLs/                 # Python 标准库扩展
    └── Lib/                  # Python 标准库和第三方包
        ├── encodings/        # 编码支持（必需）
        ├── site-packages/    # 第三方包
        │   ├── numpy/
        │   ├── sklearn/
        │   ├── scipy/
        │   ├── umap/
        │   ├── numba/
        │   └── ...
        └── ...               # 其他标准库模块
```

## 技术细节

### Python 环境初始化

程序启动时会自动：

1. 通过 `std::env::current_exe()` 获取可执行文件位置
2. 设置 `PYTHONHOME` 环境变量指向 `./python` 目录
3. 将 `site-packages` 添加到 Python 的 `sys.path`

### 包含的 Python 包

打包的 Python 环境包含以下包：

- **numpy** - 数值计算
- **scikit-learn (sklearn)** - 机器学习算法
- **scipy** - 科学计算
- **umap-learn** - 降维算法
- **numba** - JIT 编译（UMAP 依赖）
- **llvmlite** - LLVM 绑定（numba 依赖）
- **joblib** - 并行计算（sklearn 依赖）
- **pynndescent** - 最近邻搜索（UMAP 依赖）
- **threadpoolctl** - 线程池控制
- **colorama** - 跨平台颜色输出

### 体积优化

打包过程中会自动：

1. **精简标准库** - 只复制必需的模块（os, sys, numpy 依赖等）
2. **移除测试文件** - 删除所有 test/, tests/, *_test.py 文件
3. **清理缓存** - 删除 __pycache__ 和 .pyc/.pyo 文件
4. **排除文档** - 不复制文档和示例代码

## 运行要求

- **操作系统**: Windows 10/11 (64-bit)
- **运行时**: 已包含 MSVC 运行时 (vcruntime140.dll)
- **其他**: 无需安装 Python，所有依赖已打包

## 故障排除

### 问题 1：找不到 Python 模块

如果运行时报错 "No module named 'xxx'"：

1. 检查 `dist/treedect/python/Lib/site-packages/` 是否存在该模块
2. 确保模块在 `build.rs` 的 `required_packages` 列表中
3. 重新运行打包脚本

### 问题 2：DLL 加载失败

如果报错缺少 DLL：

1. 检查所有核心 DLL 是否在 `dist/treedect/` 目录
2. 检查目标机器是否安装了 VC++ Redistributable
3. 尝试将所有 DLL 复制到与 .exe 同级目录

### 问题 3：程序闪退

1. 从命令行运行查看错误信息：
   ```powershell
   cd dist\treedect
   .\treedect.exe
   ```
2. 检查日志输出
3. 确保 Python 环境完整（python/ 目录存在且包含 Lib/ 和 DLLs/）

### 调试模式

如需查看 Python 初始化信息，可以设置环境变量：

```powershell
$env:RUST_LOG="debug"
.\treedect.exe
```

## 更新 Python 包

如果需要更新 Python 包：

1. 激活 pyenv 环境：
   ```powershell
   pyenv\install\python.exe -m pip install --upgrade numpy sklearn umap-learn
   ```

2. 重新打包：
   ```powershell
   .\package.ps1 -Clean
   ```

## 许可证

打包的 Python 环境包含以下组件：

- Python: PSF License
- numpy/scipy/sklearn: BSD License
- umap-learn: BSD License
- numba: BSD License

请确保遵守各组件的许可证要求。
