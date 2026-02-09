#[cfg(test)]
mod tests {

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

    #[test]
    fn test_dilation() {
        use image::{ImageBuffer, Luma};
        use imageproc::distance_transform::Norm;
        use imageproc::morphology;
        let img_path = r"C:\Users\Activator\code\treedect\depth.png";
        let image: ImageBuffer<Luma<u8>, Vec<u8>> = image::ImageReader::open(img_path)
            .unwrap()
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .to_luma8();
        let dilated_image = morphology::grayscale_dilate(&image, &morphology::Mask::diamond(50));
        let dilated_path = r"C:\Users\Activator\code\treedect\depth_dilated.png";
        dilated_image.save(dilated_path).unwrap();

        // 逐元素相减生成diff图片
        let mut sampling_points = Vec::new();
        let mut diff_image = ImageBuffer::new(image.width(), image.height());
        for (x, y, pixel) in image.enumerate_pixels() {
            let original_val = pixel[0] as i32;
            let dilated_val = dilated_image.get_pixel(x, y)[0] as i32;
            let diff_val = (dilated_val - original_val).max(0).min(255) as u8;
            if diff_val == 0 {
                sampling_points.push((x, y));
            }
            diff_image.put_pixel(x, y, Luma([diff_val]));
        }

        // 保存diff图片
        let diff_path = r"C:\Users\Activator\code\treedect\depth_diff.png";
        diff_image.save(diff_path).unwrap();
        println!("sampling_points: {:?}", sampling_points);
    }

    #[test]
    fn test_simple_dilation() {
        use image::GrayImage;
        use image::{ImageBuffer, Luma};
        use imageproc::distance_transform::Norm;
        use imageproc::gray_image;
        use imageproc::morphology;
        let image = gray_image!(
            0,   0,   0,   0,   0;
            0,   0,   0,   0,   0;
            0,   0, 200,   0,   0;
            0,   0,   0,   0,   0;
            0,   0,   0,   0,   0
        );
        let dilated_image = morphology::dilate(&image, Norm::L2, 2);
        println!("{:?}", dilated_image);
    }

    #[test]
    fn test_dinov2() {
        use ndarray::Array;
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;
        use ort::execution_providers::DirectMLExecutionProvider;
        use ort::session::Session;
        use ort::session::builder::GraphOptimizationLevel;
        use ort::value::TensorRef;

        let x = Array::random((2, 3, 448, 448), Normal::new(0.0, 1.0).unwrap()).mapv(|x| x as f32);

        let model_path = r"C:\Users\chensy\code\treedect\output_models\";
        let onnx_path = std::path::Path::new(model_path)
            .join("dinov2_vitb_reg.onnx")
            .to_string_lossy()
            .to_string();
        log::info!("Initializing ONNX Runtime with execution provider");
        let mut session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file(onnx_path)
            .unwrap();

        let input_tensor = TensorRef::from_array_view(&x).unwrap();
        let output = session.run(ort::inputs!["img" => input_tensor]).unwrap();
        let patch_tokens = &output["patch_tokens"];
        let patch_tokens = patch_tokens.try_extract_array::<f32>().unwrap();
        assert!(!patch_tokens.is_any_nan());
    }
}
