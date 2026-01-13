// #[cfg(test)]
// mod tests {
//     #[test]
//     fn test_ort() {
//         use ndarray::{Array2, Array3, Array4, ArrayView2, Axis, array, s};
//         use numpy::{IntoPyArray, PyArray, PyArray4, PyArrayMethods, ToPyArray};
//         use pyo3::prelude::*;
//         let sam2_onnx = Python::attach(|py| {
//             // Add onnxinfer directory to Python path
//             let sys_path = py.import("sys")?.getattr("path")?;
//             sys_path.call_method1("append", ("onnxinfer",))?;

//             // Import sam2onnx module and create SAM2ONNX instance
//             let sam2onnx_module = py.import("sam2onnx")?;
//             let sam2onnx_class = sam2onnx_module.getattr("SAM2ONNX")?;
//             let sam2_onnx = sam2onnx_class.call1((
//                 r"C:\Users\Activator\code\treedect\output_models\sam2_hiera_small.encoder.onnx",
//                 r"C:\Users\Activator\code\treedect\output_models\sam2_hiera_small.decoder.onnx",
//             ))?;

//             log::info!("Successfully loaded Python SAM2ONNX model");
//             Ok::<Py<PyAny>, Box<dyn std::error::Error>>(sam2_onnx.into())
//         })
//         .unwrap();
//         let batch = Array4::<f32>::ones((2, 3, 1024, 1024));
//         let result = Python::attach(|py| {
//             let batch = batch.into_pyarray(py);
//             sam2_onnx.call_method1(py, "encode", (batch,))
//         });
//         println!("{:?}", result);
//     }
// }
