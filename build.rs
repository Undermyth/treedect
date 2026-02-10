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

    // 复制核心 DLL 到二进制目录（便于加载）
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

    // 复制 DLLs 目录（Python 标准库扩展 - 包含 .pyd 文件）
    copy_dir_with_filter(
        &python_src.join("DLLs"),
        &python_dest.join("DLLs"),
        |path| !should_exclude_dll(path),
    );

    // 统计复制的 .pyd 文件数量
    let pyd_count = WalkDir::new(&python_dest.join("DLLs"))
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

    // 创建 Lib 目录结构
    let lib_src = python_src.join("Lib");
    let lib_dest = python_dest.join("Lib");
    fs::create_dir_all(&lib_dest).unwrap();

    // 复制精简后的标准库
    copy_minimal_stdlib(&lib_src, &lib_dest);

    // 复制必需的第三方包
    let site_packages_src = lib_src.join("site-packages");
    let site_packages_dest = lib_dest.join("site-packages");

    if site_packages_src.exists() {
        fs::create_dir_all(&site_packages_dest).unwrap();
        copy_required_packages(&site_packages_src, &site_packages_dest);
    }

    println!("cargo:warning=Python environment setup complete");
}

fn copy_dir_with_filter<F>(src: &Path, dst: &Path, filter: F)
where
    F: Fn(&Path) -> bool,
{
    if !src.exists() {
        return;
    }

    fs::create_dir_all(dst).unwrap();

    for entry in WalkDir::new(src) {
        let entry = entry.unwrap();
        let path = entry.path();

        if !filter(path) {
            continue;
        }

        let relative = path.strip_prefix(src).unwrap();
        let dest_path = dst.join(relative);

        if path.is_file() {
            fs::create_dir_all(dest_path.parent().unwrap()).unwrap();
            fs::copy(path, dest_path).unwrap();
        } else if path.is_dir() && !should_exclude_dir(path) {
            fs::create_dir_all(&dest_path).unwrap();
        }
    }
}

fn copy_dir(src: &Path, dst: &Path) {
    copy_dir_with_filter(src, dst, |_| true);
}

fn should_exclude_dll(path: &Path) -> bool {
    let path_str = path.to_string_lossy().to_lowercase();

    // 排除测试相关的 DLL
    let exclusions = vec!["_test", "test_"];

    for ex in &exclusions {
        if path_str.contains(ex) {
            return true;
        }
    }

    false
}

fn should_exclude_dir(path: &Path) -> bool {
    let path_str = path.to_string_lossy().to_lowercase();

    let exclusions = vec!["__pycache__", "test", "tests", "testing"];

    for ex in &exclusions {
        if path_str.contains(ex) {
            return true;
        }
    }

    false
}

fn copy_minimal_stdlib(src: &Path, dst: &Path) {
    // 复制标准库的核心目录，排除测试和文档
    // 策略：复制关键目录中的所有文件，而不是维护一个模块列表

    let essential_dirs = vec![
        "encodings",       // 编码支持（Python启动必需）
        "collections",     // 数据结构
        "logging",         // 日志
        "importlib",       // 导入系统
        "json",            // JSON
        "re",              // 正则表达式
        "decimal",         // 十进制运算
        "email",           // 邮件处理（logging依赖）
        "html",            // HTML处理
        "http",            // HTTP支持
        "urllib",          // URL处理
        "xml",             // XML支持
        "ctypes",          // C类型
        "multiprocessing", // 多进程
        "concurrent",      // 并发
        "asyncio",         // 异步IO
        "unittest",        // 单元测试（某些库依赖）
    ];

    // 需要完整复制的目录（不排除 test 子目录）
    let full_copy_dirs = vec!["unittest"];

    // 复制关键目录
    for dir in &essential_dirs {
        let src_path = src.join(dir);
        let dst_path = dst.join(dir);
        if src_path.exists() {
            if full_copy_dirs.contains(&dir.as_ref()) {
                // 完整复制（不排除 test 子目录）
                println!("cargo:warning=Full copying directory: {}", dir);
                copy_package_full(&src_path, &dst_path);
            } else {
                // 标准复制（排除 test 子目录）
                println!("cargo:warning=Copying directory: {}", dir);
                copy_dir_with_filter(&src_path, &dst_path, |p| {
                    !should_exclude_file(p) && !should_exclude_dir(p)
                });
            }
        }
    }

    // 复制所有 .py 文件（除去测试文件）
    // 这样确保所有模块依赖都被满足
    println!("cargo:warning=Copying all Python files from Lib directory...");
    for entry in WalkDir::new(src).max_depth(1) {
        let entry = entry.unwrap();
        let path = entry.path();

        // 跳过目录（已经处理过关键目录）
        if path.is_dir() {
            let dir_name = path.file_name().unwrap().to_string_lossy();
            // 排除特定目录
            if dir_name.starts_with("test")
                || dir_name.starts_with("__pycache__")
                || dir_name.starts_with("distutils")
                || dir_name.starts_with("ensurepip")
                || dir_name.starts_with("venv")
                || dir_name.starts_with("lib2to3")
                || dir_name.starts_with("idlelib")
                || dir_name.starts_with("tkinter")
                || dir_name.starts_with("turtledemo")
                || dir_name.starts_with("pydoc_data")
            {
                continue;
            }
            // 已经复制过的目录跳过
            if essential_dirs.contains(&dir_name.as_ref()) {
                continue;
            }
        }

        // 复制文件
        if path.is_file() && path.extension().map_or(false, |ext| ext == "py") {
            let file_name = path.file_name().unwrap().to_string_lossy();

            // 排除测试文件
            if file_name.starts_with("test")
                || file_name.contains("_test")
                || file_name.starts_with("__")
            {
                continue;
            }

            let dst_path = dst.join(&*file_name);
            fs::copy(path, dst_path).unwrap();
        }
    }

    // 特别处理：确保 encodings 目录存在且完整
    let encodings_src = src.join("encodings");
    let encodings_dst = dst.join("encodings");
    if encodings_src.exists() {
        println!("cargo:warning=Copying encodings directory...");
        copy_dir_with_filter(&encodings_src, &encodings_dst, |p| !should_exclude_file(p));

        // 验证 encodings 目录是否复制成功
        if encodings_dst.exists() {
            let count = WalkDir::new(&encodings_dst)
                .into_iter()
                .filter(|e| e.as_ref().unwrap().path().is_file())
                .count();
            println!("cargo:warning=Copied {} files to encodings/", count);
        }
    } else {
        println!(
            "cargo:warning=ERROR: encodings directory not found at {:?}",
            encodings_src
        );
    }

    // 特别处理：复制以 _ 开头的私有模块（如 _collections_abc.py）
    for entry in WalkDir::new(src).max_depth(1) {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_file() {
            let file_name = path.file_name().unwrap().to_string_lossy();
            if file_name.starts_with("_") && file_name.ends_with(".py") {
                let dst_path = dst.join(&*file_name);
                if !dst_path.exists() {
                    fs::copy(path, dst_path).unwrap();
                }
            }
        }
    }

    // 统计复制的文件数量
    let py_count = WalkDir::new(dst)
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

fn should_exclude_file(path: &Path) -> bool {
    let path_str = path.to_string_lossy().to_lowercase();

    // 排除测试文件和缓存
    if path_str.contains("__pycache__") {
        return true;
    }

    if path_str.ends_with(".pyc") || path_str.ends_with(".pyo") {
        return true;
    }

    // 排除测试文件，但保留运行必需的测试相关模块
    // 注意：某些包虽然名字里有 test，但不是测试文件，而是运行时必需的
    if path_str.contains("_test") || path_str.contains("test_") {
        // 例外：保留运行必需的测试工具模块
        let runtime_test_modules = [
            "_testutils",
            "_tester",
            "numpy.testing",
            "scipy._lib._testutils",
        ];
        for module in &runtime_test_modules {
            if path_str.contains(module) {
                return false;
            }
        }
        return true;
    }

    false
}

// 完整复制一个包（只排除 __pycache__ 和 .pyc/.pyo 文件，不排除 test 目录）
fn copy_package_full(src: &Path, dst: &Path) {
    if !src.exists() {
        return;
    }

    fs::create_dir_all(dst).unwrap();

    for entry in WalkDir::new(src) {
        let entry = entry.unwrap();
        let path = entry.path();

        // 只排除 __pycache__ 和 .pyc/.pyo 文件
        let path_str = path.to_string_lossy().to_lowercase();
        if path_str.contains("__pycache__")
            || path_str.ends_with(".pyc")
            || path_str.ends_with(".pyo")
        {
            continue;
        }

        let relative = path.strip_prefix(src).unwrap();
        let dest_path = dst.join(relative);

        if path.is_file() {
            fs::create_dir_all(dest_path.parent().unwrap()).unwrap();
            fs::copy(path, dest_path).unwrap();
        } else if path.is_dir() {
            fs::create_dir_all(&dest_path).unwrap();
        }
    }
}

fn copy_required_packages(src: &Path, dst: &Path) {
    // 需要完整复制的关键包（科学计算库，依赖关系复杂）
    let full_copy_packages = vec!["numpy", "sklearn", "scipy", "umap"];

    // 其他第三方包（可以使用过滤）
    let other_packages = vec![
        "numba",
        "llvmlite",
        "joblib",
        "pynndescent",
        "threadpoolctl",
        "colorama",
    ];

    // 完整复制关键包（只排除 __pycache__ 和 .pyc 文件，不排除 test 目录）
    for package in &full_copy_packages {
        let pkg_src = src.join(package);
        if pkg_src.exists() {
            let pkg_dst = dst.join(package);
            println!("cargo:warning=Full copying package: {}", package);
            // 使用专门的函数复制，不排除 test 目录
            copy_package_full(&pkg_src, &pkg_dst);
        } else {
            println!("cargo:warning=Warning: Package '{}' not found", package);
        }
    }

    // 复制其他包（使用标准过滤）
    for package in &other_packages {
        let pkg_src = src.join(package);
        if pkg_src.exists() {
            let pkg_dst = dst.join(package);
            println!("cargo:warning=Copying package: {}", package);
            copy_dir_with_filter(&pkg_src, &pkg_dst, |p| !should_exclude_file(p));
        }
    }

    // 复制所有包的 .libs/.lib/_libs 目录（DLL 依赖）
    let all_packages: Vec<&str> = full_copy_packages
        .iter()
        .chain(other_packages.iter())
        .copied()
        .collect();

    for package in &all_packages {
        let lib_patterns = vec![
            format!("{}.libs", package),
            format!("{}.lib", package),
            format!("{}_libs", package),
        ];

        for lib_pattern in &lib_patterns {
            let libs_src = src.join(lib_pattern);
            if libs_src.exists() {
                let libs_dst = dst.join(lib_pattern);
                println!("cargo:warning=  Copying libs: {}", lib_pattern);
                copy_dir_with_filter(&libs_src, &libs_dst, |_| true);
            }
        }

        // 复制 .dist-info 目录
        let dist_info_name = format!("{}-", package.replace("_", "-"));
        for entry in fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&dist_info_name) && name.ends_with(".dist-info") {
                let dist_dst = dst.join(&name);
                copy_dir_with_filter(&entry.path(), &dist_dst, |_| true);
            }
        }
    }
}
