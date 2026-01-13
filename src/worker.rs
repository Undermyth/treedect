use ipc_channel::ipc;
use ort::execution_providers::DirectMLExecutionProvider;
use phf::{Map, phf_map};
use serde::{Deserialize, Serialize};

use crate::models::dinov2;
use crate::models::sam2;

static MODEL2FILENAME: Map<&'static str, &'static str> = phf_map! {
    "sam2_small" => "sam2_hiera_small",
    "dinov2_base" => "dinov2_vitb_reg",
};

#[derive(Serialize, Deserialize)]
pub struct SegmentTask {
    pub memory_addr: String,
}

#[derive(Serialize, Deserialize)]
pub struct SegmentResponse {
    pub memory_addr: String,
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
                                segment_model = Some(model);
                                let _result = worker_channel.load_res_tx.send(true);
                            }
                            Err(e) => {
                                log::error!("Error {}", e);
                                let _result = worker_channel.load_res_tx.send(false);
                            }
                        }
                    }
                    ModelType::Feature => {
                        let model_prefix = MODEL2FILENAME.get(&task.model_name).unwrap();
                        let model_path = std::path::Path::new(&task.model_path)
                            .join(format!("{}.onnx", model_prefix))
                            .to_string_lossy()
                            .to_string();
                        let load_result = dinov2::Dinov2Model::from_path(&model_path);
                        match load_result {
                            Ok(model) => {
                                classify_model = Some(model);
                                let _result = worker_channel.load_res_tx.send(true);
                            }
                            Err(e) => {
                                log::error!("Error {}", e);
                                let _result = worker_channel.load_res_tx.send(false);
                            }
                        }
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
    }
}
