use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use std::env;
use std::ffi::CString;
use std::path::PathBuf;

pub struct SegmentFeatures {
    pub segment_ids: Vec<usize>,
    pub features: Array2<f32>,
    pub areas: Vec<usize>,
}

// 嵌入 Python 代码
const CLUSTER_PYTHON_CODE: &str = include_str!("cluster.py");

/// 确保 site-packages 在 Python 路径中
fn ensure_site_packages(py: Python) {
    let exe_path = env::current_exe().expect("Failed to get executable path");
    let exe_dir = exe_path
        .parent()
        .expect("Failed to get executable directory");
    let site_packages = exe_dir.join("python").join("Lib").join("site-packages");

    if site_packages.exists() {
        let sys = py.import("sys").expect("Failed to import sys");
        let path = sys.getattr("path").expect("Failed to get sys.path");
        let path_str = site_packages.to_string_lossy().to_string();

        // // 检查是否已在路径中
        // let contains = path
        //     .try_iter()
        //     .unwrap()
        //     .any(|p: PyResult<Bound<'_, PyAny>>| {
        //         if let Ok(p_obj) = p {
        //             if let Ok(p_str) = p_obj.extract::<String>() {
        //                 p_str == path_str
        //             } else {
        //                 false
        //             }
        //         } else {
        //             false
        //         }
        //     });

        // if !contains {
        path.call_method1("insert", (0usize, path_str.as_str()))
            .expect("Failed to add site-packages to path");
        log::debug!(
            "Added site-packages to Python path: {}",
            site_packages.display()
        );
        // }
    }
}

pub fn cluster(features: SegmentFeatures, n_classes: usize) -> Vec<Vec<usize>> {
    Python::with_gil(|py| {
        // 确保 site-packages 在路径中（主程序已设置 PYTHONHOME，这里只是双重保险）
        ensure_site_packages(py);
        // 使用嵌入的 Python 代码创建模块
        let cluster_module = PyModule::from_code(
            py,
            CString::new(CLUSTER_PYTHON_CODE).unwrap().as_c_str(),
            c_str!("cluster.py"),
            c_str!("cluster"),
        )
        .expect("Failed to create cluster module from embedded code");

        // 获取 cluster 函数
        let cluster_func = cluster_module
            .getattr("cluster")
            .expect("Failed to get cluster function");

        // 将 segment_ids 转换为 numpy 数组
        let ids_array = features
            .segment_ids
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>();
        let ids_py = PyArray1::from_vec(py, ids_array);

        // 将 features 转换为 numpy 数组 [N, d]
        let features_py = PyArray2::from_array(py, &features.features);

        // 将 areas 转换为 numpy 数组并 reshape 为 [N, 1]
        let areas_array = features
            .areas
            .iter()
            .map(|&x| x as f64)
            .collect::<Vec<f64>>();
        let areas_1d = PyArray1::from_vec(py, areas_array);
        let areas_py = areas_1d
            .reshape([features.areas.len(), 1])
            .expect("Failed to reshape areas array");

        // 调用 Python 的 cluster 函数
        let result = cluster_func
            .call1((ids_py, features_py, areas_py, n_classes as i64))
            .expect("Failed to call cluster function");

        // 解析返回结果：List[np.ndarray]
        let result_list = result
            .cast::<PyList>()
            .expect("Expected a list of numpy arrays");

        // 转换为 Vec<Vec<usize>>
        let mut clusters: Vec<Vec<usize>> = Vec::with_capacity(n_classes);
        for i in 0..result_list.len() {
            let item = result_list.get_item(i).expect("Failed to get cluster item");
            let cluster_array = item.cast::<PyArray1<i64>>().expect("Expected numpy array");

            let readonly_array = cluster_array.readonly();
            let slice = readonly_array
                .as_slice()
                .expect("Failed to get array slice");

            let cluster_ids: Vec<usize> = slice.iter().map(|&x| x as usize).collect();

            clusters.push(cluster_ids);
        }

        clusters
    })
}
