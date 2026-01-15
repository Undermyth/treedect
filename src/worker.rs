use std::error::Error;

use ipc_channel::ipc;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::ArrayView4;
use ort::execution_providers::DirectMLExecutionProvider;
use phf::{Map, phf_map};
use serde::{Deserialize, Serialize};
use shared_memory::*;

use crate::models::batcher;
use crate::models::dinov2;
use crate::models::sam2;

static MODEL2FILENAME: Map<&'static str, &'static str> = phf_map! {
    "sam2_small" => "sam2_hiera_small",
    "dinov2_base" => "dinov2_vitb_reg",
};

#[derive(Serialize, Deserialize)]
pub struct SegmentTask {
    pub batch_buffer_name: String,
    pub coord_buffer_name: String,
    pub sample_coord_buffer_name: String,
    pub batch_size: usize,
    pub segment_rel: usize,
    pub original_size: usize,
}

#[derive(Serialize, Deserialize)]
pub struct SegmentResponse {
    pub mask_buffer_name: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum ModelType {
    Segment,
    Feature,
    Depth,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LoadTask {
    pub model_path: String,
    pub model_type: ModelType,
    pub model_name: String,
}

#[derive(Serialize, Deserialize)]
pub struct GUIChannel {
    pub load_req_tx: ipc::IpcSender<LoadTask>,
    pub load_res_rx: ipc::IpcReceiver<bool>,
    pub segment_req_tx: ipc::IpcSender<SegmentTask>,
    pub segment_res_rx: ipc::IpcReceiver<SegmentResponse>,
}

#[derive(Serialize, Deserialize)]
pub struct WorkerChannel {
    pub load_req_rx: ipc::IpcReceiver<LoadTask>,
    pub load_res_tx: ipc::IpcSender<bool>,
    pub segment_req_rx: ipc::IpcReceiver<SegmentTask>,
    pub segment_res_tx: ipc::IpcSender<SegmentResponse>,
}

pub fn main(server_name: &str) {
    // use ndarray::{Array4, ArrayView2, Axis, array, s};
    // use ort::execution_providers::DirectMLExecutionProvider;
    // use ort::session::Session;
    // use ort::session::builder::GraphOptimizationLevel;
    // use ort::value::TensorRef;
    // let _result = ort::init()
    //     .with_execution_providers([DirectMLExecutionProvider::default()
    //         .build()
    //         .error_on_failure()])
    //     .commit();
    // let model_prefix = "sam2_hiera_small";
    // let model_path = r"C:\Users\Activator\code\treedect\output_models\";
    // let encoder_path = std::path::Path::new(model_path)
    //     .join(format!("{}.encoder.onnx", model_prefix))
    //     .to_string_lossy()
    //     .to_string();
    // let decoder_path = std::path::Path::new(model_path)
    //     .join(format!("{}.decoder.onnx", model_prefix))
    //     .to_string_lossy()
    //     .to_string();
    // let mut model = sam2::SAM2Model::from_path(&encoder_path, &decoder_path).unwrap();
    // let batch = Array4::<f32>::ones((2, 3, 1024, 1024));
    // let input_tensor = TensorRef::from_array_view(&batch).unwrap();
    // let batch = batcher::SAM2Batch {
    //     image: batch.to_owned(),
    //     coordinates: Array2::<usize>::ones((2, 2)),
    //     sampling_coords: Array2::<f32>::ones((2, 2)),
    //     original_size: 1024,
    // };
    // let output = model.forward(batch).unwrap();
    log::info!("Worker connecting to server: {}", server_name);

    // Connect to the parent's one-shot server
    let parent_sender: ipc::IpcSender<GUIChannel> =
        ipc::IpcSender::connect(server_name.to_string()).unwrap();

    // Create the actual channels for ongoing communication
    let (load_req_tx, load_req_rx) = ipc::channel::<LoadTask>().unwrap();
    let (load_res_tx, load_res_rx) = ipc::channel::<bool>().unwrap();
    let (segment_req_tx, segment_req_rx) = ipc::channel::<SegmentTask>().unwrap();
    let (segment_res_tx, segment_res_rx) = ipc::channel::<SegmentResponse>().unwrap();

    // Send the channels to the parent as the first (and only) message
    let worker_channel = WorkerChannel {
        load_req_rx,
        load_res_tx,
        segment_req_rx,
        segment_res_tx,
    };

    // Now set up the worker with the channels it will use
    let gui_channel = GUIChannel {
        load_req_tx,
        load_res_rx,
        segment_req_tx,
        segment_res_rx,
    };
    parent_sender.send(gui_channel).unwrap();

    let mut segment_model: Option<sam2::SAM2Model> = None;
    let mut classify_model: Option<dinov2::Dinov2Model> = None;

    log::info!("Initializing ONNX Runtime with execution provider");
    let _result = ort::init()
        .with_execution_providers([DirectMLExecutionProvider::default()
            .build()
            .error_on_failure()])
        .commit();

    loop {
        match worker_channel.load_req_rx.try_recv() {
            Ok(task) => {
                // Handle the load task
                match task.model_type {
                    ModelType::Segment => {
                        segment_model =
                            Some(worker_load_segment_model(task, &worker_channel).unwrap());
                    }
                    ModelType::Feature => {
                        classify_model =
                            Some(worker_load_classify_model(task, &worker_channel).unwrap());
                    }
                    ModelType::Depth => {
                        let _result = worker_channel.load_res_tx.send(false);
                    }
                }
            }
            Err(ipc::TryRecvError::Empty) => {}
            Err(ipc::TryRecvError::IpcError(error)) => {
                log::error!("Error in IPC communication: {}", error);
                log::info!("Parent process appears to have closed, exiting worker");
                break;
            }
        }
        match worker_channel.segment_req_rx.try_recv() {
            Ok(task) => {
                worker_segment(task, &worker_channel, segment_model.as_mut().unwrap());
            }
            Err(ipc::TryRecvError::Empty) => {}
            Err(ipc::TryRecvError::IpcError(error)) => {
                log::error!("Error in IPC communication: {}", error);
                log::info!("Parent process appears to have closed, exiting worker");
                break;
            }
        }
    }
}

pub fn get_shm(bytes: usize, name: &str) -> Result<Shmem, ShmemError> {
    let shm = match ShmemConf::new().size(bytes).flink(name).create() {
        Ok(m) => m,
        Err(ShmemError::LinkExists) => ShmemConf::new().flink(name).open().unwrap(),
        Err(e) => {
            eprintln!("Unable to create or open shmem flink {name} : {e}");
            return Err(e);
        }
    };
    Ok(shm)
}

fn worker_load_segment_model(
    task: LoadTask,
    worker_channel: &WorkerChannel,
) -> Result<sam2::SAM2Model, Box<dyn Error>> {
    let model_prefix = MODEL2FILENAME.get(&task.model_name).unwrap();
    let encoder_path = std::path::Path::new(&task.model_path)
        .join(format!("{}.encoder.onnx", model_prefix))
        .to_string_lossy()
        .to_string();
    let decoder_path = std::path::Path::new(&task.model_path)
        .join(format!("{}.decoder.onnx", model_prefix))
        .to_string_lossy()
        .to_string();
    let load_result = sam2::SAM2Model::from_path(&encoder_path, &decoder_path);
    match load_result {
        Ok(model) => {
            let _result = worker_channel.load_res_tx.send(true);
            return Ok(model);
        }
        Err(e) => {
            log::error!("Error {}", e);
            let _result = worker_channel.load_res_tx.send(false);
            return Err(e);
        }
    }
}

fn worker_load_classify_model(
    task: LoadTask,
    worker_channel: &WorkerChannel,
) -> Result<dinov2::Dinov2Model, Box<dyn Error>> {
    let model_prefix = MODEL2FILENAME.get(&task.model_name).unwrap();
    let model_path = std::path::Path::new(&task.model_path)
        .join(format!("{}.onnx", model_prefix))
        .to_string_lossy()
        .to_string();
    let load_result = dinov2::Dinov2Model::from_path(&model_path);
    match load_result {
        Ok(model) => {
            let _result = worker_channel.load_res_tx.send(true);
            return Ok(model);
        }
        Err(e) => {
            log::error!("Error {}", e);
            let _result = worker_channel.load_res_tx.send(false);
            return Err(e);
        }
    }
}

fn worker_segment(
    task: SegmentTask,
    worker_channel: &WorkerChannel,
    segment_model: &mut sam2::SAM2Model,
) {
    let batch_shm = ShmemConf::new()
        .flink(task.batch_buffer_name)
        .open()
        .unwrap();
    let coord_shm = ShmemConf::new()
        .flink(task.coord_buffer_name)
        .open()
        .unwrap();
    let sample_coord_shm = ShmemConf::new()
        .flink(task.sample_coord_buffer_name)
        .open()
        .unwrap();
    let batch_shape = (task.batch_size, 3, task.segment_rel, task.segment_rel);
    let coord_shape = (task.batch_size, 2);
    let sample_coord_shape = (task.batch_size, 2);
    unsafe {
        let batch_ptr = batch_shm.as_ptr() as *mut f32;
        let coord_ptr = coord_shm.as_ptr() as *mut usize;
        let sample_coord_ptr = sample_coord_shm.as_ptr() as *mut f32;
        let batch = ArrayView4::from_shape_ptr(batch_shape, batch_ptr);
        let coord = ArrayView2::from_shape_ptr(coord_shape, coord_ptr);
        let sample_coord = ArrayView2::from_shape_ptr(sample_coord_shape, sample_coord_ptr);
        use ndarray::{Array4, ArrayView2, Axis, array, s};
        log::info!("{:?}", batch_shape);
        let batch = Array4::<f32>::ones((2, 3, 1024, 1024));
        let batch = batcher::SAM2Batch {
            image: batch.to_owned(),
            coordinates: coord.to_owned(),
            sampling_coords: sample_coord.to_owned(),
            original_size: task.original_size,
        };
        let result = segment_model.forward(batch).unwrap();
        let mask_byte = result.mask_logits.len() * std::mem::size_of::<f32>();
        let result_shm = get_shm(mask_byte, "result_buffer").unwrap();
        let shm_ptr = result_shm.as_ptr();
        let mask_ptr = result.mask_logits.as_ptr();
        std::ptr::copy_nonoverlapping(mask_ptr, shm_ptr as *mut f32, mask_byte);
        let _result = worker_channel.segment_res_tx.send(SegmentResponse {
            mask_buffer_name: "result_buffer".to_string().to_string(),
        });
    }
}
