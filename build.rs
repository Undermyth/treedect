use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile = env::var("PROFILE").unwrap();
    let target_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with(&profile))
        .unwrap()
        .to_path_buf();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=pyenv/install");
    println!("cargo:rerun-if-changed=src/models/cluster.py");

    // 复制 Python 环境到输出目录
    copy_python_env(&target_dir);

    // 设置运行时环境变量
    println!("cargo:rustc-env=PYTHONHOME=python");
}

fn copy_python_env(target_dir: &Path) {
    let python_src = PathBuf::from("pyenv/install");
    let python_dest = target_dir.join("python");

    if !python_src.exists() {
        panic!("Python environment not found at pyenv/install");
    }

    // 如果目标目录已存在，先删除
    if python_dest.exists() {
        fs::remove_dir_all(&python_dest).unwrap();
    }

    // 创建目标目录
    fs::create_dir_all(&python_dest).unwrap();

    println!(
        "cargo:warning=Copying Python environment to {}",
        python_dest.display()
    );

    // 1. 复制核心 DLL 到二进制目录（便于加载）
    copy_core_dlls(&python_src, target_dir);

    // 2. 复制 DLLs 目录（Python 扩展模块 .pyd 文件）
    copy_dlls_directory(&python_src, &python_dest);

    // 3. 复制 Lib 目录（Python 标准库）
    copy_lib_directory(&python_src, &python_dest);

    // 4. 复制 site-packages（第三方包）
    copy_site_packages(&python_src, &python_dest);

    println!("cargo:warning=Python environment setup complete");
}

/// 复制核心 DLL 文件到二进制目录
fn copy_core_dlls(python_src: &Path, target_dir: &Path) {
    let core_dlls = vec![
        "python311.dll",
        "python3.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    ];

    for dll in &core_dlls {
        let src = python_src.join(dll);
        let dest = target_dir.join(dll);
        if src.exists() {
            fs::copy(&src, &dest).unwrap();
            println!("cargo:warning=Copied {} to binary directory", dll);
        } else {
            println!("cargo:warning=Warning: {} not found", dll);
        }
    }
}

/// 复制 DLLs 目录（包含 .pyd 扩展模块）
fn copy_dlls_directory(python_src: &Path, python_dest: &Path) {
    let dlls_src = python_src.join("DLLs");
    let dlls_dest = python_dest.join("DLLs");

    if dlls_src.exists() {
        println!("cargo:warning=Copying DLLs directory...");
        copy_directory_simple(&dlls_src, &dlls_dest);

        // 统计复制的 .pyd 文件
        let pyd_count = WalkDir::new(&dlls_dest)
            .into_iter()
            .filter(|e| {
                if let Ok(entry) = e {
                    entry.path().extension().map_or(false, |ext| ext == "pyd")
                } else {
                    false
                }
            })
            .count();
        println!("cargo:warning=Copied {} .pyd files to DLLs/", pyd_count);
    }
}

/// 复制 Lib 目录（Python 标准库）
fn copy_lib_directory(python_src: &Path, python_dest: &Path) {
    let lib_src = python_src.join("Lib");
    let lib_dest = python_dest.join("Lib");

    if !lib_src.exists() {
        println!("cargo:warning=Warning: Lib directory not found");
        return;
    }

    fs::create_dir_all(&lib_dest).unwrap();
    println!("cargo:warning=Copying Lib directory...");

    // 遍历 Lib 目录下的所有项目
    for entry in fs::read_dir(&lib_src).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        // 跳过 site-packages（单独处理）
        if name == "site-packages" {
            continue;
        }

        // 跳过已知不需要的目录
        if path.is_dir() {
            let skip_dirs = [
                "__pycache__",
                "test",
                "tests",
                "idlelib",
                "tkinter",
                "turtledemo",
                "pydoc_data",
                "distutils",
                "ensurepip",
                "venv",
                "lib2to3",
            ];

            if skip_dirs.contains(&name.as_str()) {
                continue;
            }
        }

        let dest_path = lib_dest.join(&name);

        if path.is_file() {
            // 复制 .py 文件
            if path.extension().map_or(false, |ext| ext == "py") {
                fs::copy(&path, &dest_path).unwrap();
            }
        } else if path.is_dir() {
            // 递归复制目录
            copy_directory_simple(&path, &dest_path);
        }
    }

    // 统计复制的文件数
    let py_count = WalkDir::new(&lib_dest)
        .into_iter()
        .filter(|e| {
            if let Ok(entry) = e {
                entry.path().extension().map_or(false, |ext| ext == "py")
            } else {
                false
            }
        })
        .count();
    println!("cargo:warning=Copied {} Python files to Lib/", py_count);
}

/// 复制 site-packages 目录
fn copy_site_packages(python_src: &Path, python_dest: &Path) {
    let site_src = python_src.join("Lib").join("site-packages");
    let site_dest = python_dest.join("Lib").join("site-packages");

    if !site_src.exists() {
        println!("cargo:warning=Warning: site-packages not found");
        return;
    }

    fs::create_dir_all(&site_dest).unwrap();
    println!("cargo:warning=Copying site-packages...");

    // 遍历 site-packages 下的所有项目
    for entry in fs::read_dir(&site_src).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        let dest_path = site_dest.join(&name);

        if path.is_file() {
            // 复制所有文件（包括单文件模块如 threadpoolctl.py）
            if should_copy_file(&path) {
                fs::copy(&path, &dest_path).unwrap();
            }
        } else if path.is_dir() {
            // 复制所有目录（包括包目录如 numpy, scipy, tqdm 等）
            if should_copy_dir(&name) {
                println!("cargo:warning=  Copying: {}", name);
                copy_directory_simple(&path, &dest_path);
            }
        }
    }

    // 统计复制的包数
    let (dir_count, file_count) = count_site_packages(&site_dest);
    println!(
        "cargo:warning=Copied {} packages/directories and {} single-file modules",
        dir_count, file_count
    );
}

/// 简单复制目录 - 只排除 __pycache__ 和编译后的 .pyc/.pyo 文件
fn copy_directory_simple(src: &Path, dst: &Path) {
    if !src.exists() {
        return;
    }

    fs::create_dir_all(dst).unwrap();

    for entry in WalkDir::new(src) {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        let relative = path.strip_prefix(src).unwrap();
        let dest_path = dst.join(relative);

        if path.is_file() {
            // 检查是否应该复制此文件
            if !should_copy_file(path) {
                continue;
            }

            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::copy(path, dest_path).unwrap();
        } else if path.is_dir() {
            // 检查目录名
            let dir_name = path.file_name().unwrap_or_default().to_string_lossy();
            if dir_name == "__pycache__" {
                continue;
            }

            fs::create_dir_all(&dest_path).unwrap();
        }
    }
}

/// 判断是否应该复制文件 - 只排除编译缓存文件
fn should_copy_file(path: &Path) -> bool {
    let path_str = path.to_string_lossy().to_lowercase();

    // 排除 __pycache__ 目录中的文件
    if path_str.contains("__pycache__") {
        return false;
    }

    // 排除 .pyc 和 .pyo 文件
    if let Some(ext) = path.extension() {
        let ext = ext.to_string_lossy().to_lowercase();
        if ext == "pyc" || ext == "pyo" {
            return false;
        }
    }

    true
}

/// 判断是否应该复制目录
fn should_copy_dir(name: &str) -> bool {
    // 只排除 __pycache__ 目录
    name != "__pycache__"
}

/// 统计 site-packages 中的包数量
fn count_site_packages(site_dir: &Path) -> (usize, usize) {
    let mut dir_count = 0;
    let mut file_count = 0;

    if let Ok(entries) = fs::read_dir(site_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();

                if path.is_dir() {
                    // 统计包目录（排除 .dist-info 和 .libs 等元数据目录）
                    if !name.ends_with(".dist-info")
                        && !name.ends_with(".libs")
                        && !name.ends_with(".lib")
                        && !name.ends_with("_libs")
                    {
                        dir_count += 1;
                    }
                } else if path.is_file() && name.ends_with(".py") {
                    // 统计单文件模块
                    file_count += 1;
                }
            }
        }
    }

    (dir_count, file_count)
}
