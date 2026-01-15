#[cfg(test)]
mod tests {
    // #[test]
    // fn test_ort() {
    //     use ndarray::{Array2, Array3, Array4, ArrayView2, Axis, array, s};
    //     use numpy::{IntoPyArray, PyArray, PyArray4, PyArrayMethods, ToPyArray};
    //     use pyo3::prelude::*;
    //     let sam2_onnx = Python::attach(|py| {
    //         // Add onnxinfer directory to Python path
    //         let sys_path = py.import("sys")?.getattr("path")?;
    //         sys_path.call_method1("append", ("onnxinfer",))?;

    //         // Import sam2onnx module and create SAM2ONNX instance
    //         let sam2onnx_module = py.import("sam2onnx")?;
    //         let sam2onnx_class = sam2onnx_module.getattr("SAM2ONNX")?;
    //         let sam2_onnx = sam2onnx_class.call1((
    //             r"C:\Users\Activator\code\treedect\output_models\sam2_hiera_small.encoder.onnx",
    //             r"C:\Users\Activator\code\treedect\output_models\sam2_hiera_small.decoder.onnx",
    //         ))?;

    //         log::info!("Successfully loaded Python SAM2ONNX model");
    //         Ok::<Py<PyAny>, Box<dyn std::error::Error>>(sam2_onnx.into())
    //     })
    //     .unwrap();
    //     let batch = Array4::<f32>::ones((2, 3, 1024, 1024));
    //     let result = Python::attach(|py| {
    //         let batch = batch.into_pyarray(py);
    //         sam2_onnx.call_method1(py, "encode", (batch,))
    //     });
    //     println!("{:?}", result);
    // }

    #[test]
    fn test_rust_ort() {
        use ndarray::{Array4, ArrayView2, Axis, array, s};
        use ort::execution_providers::DirectMLExecutionProvider;
        use ort::session::Session;
        use ort::session::builder::GraphOptimizationLevel;
        use ort::value::TensorRef;
        let _result = ort::init()
            .with_execution_providers([DirectMLExecutionProvider::default()
                .build()
                .error_on_failure()])
            .commit();
        let model_prefix = "sam2_hiera_small";
        let model_path = r"C:\Users\Activator\code\treedect\output_models\";
        let encoder_path = std::path::Path::new(model_path)
            .join(format!("{}.encoder.onnx", model_prefix))
            .to_string_lossy()
            .to_string();
        let decoder_path = std::path::Path::new(model_path)
            .join(format!("{}.decoder.onnx", model_prefix))
            .to_string_lossy()
            .to_string();
        let mut encoder_session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file(encoder_path)
            .unwrap();
        log::info!("Loading decoder model from: {}", decoder_path);
        let decoder_session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file(decoder_path)
            .unwrap();
        let batch = Array4::<f32>::ones((8, 3, 1024, 1024));
        let input_tensor = TensorRef::from_array_view(&batch).unwrap();
        let output = encoder_session
            .run(ort::inputs!["image" => input_tensor])
            .unwrap();
    }
}
