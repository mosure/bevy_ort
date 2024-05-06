use std::io::ErrorKind;
use std::sync::{Arc, Mutex};

use bevy::{
    prelude::*,
    asset::{
        AssetLoader,
        AsyncReadExt,
        LoadContext,
        io::Reader,
    },
    utils::BoxedFuture,
};
use ort::{
    CoreMLExecutionProvider,
    CPUExecutionProvider,
    DirectMLExecutionProvider,
    CUDAExecutionProvider,
    GraphOptimizationLevel,
    OpenVINOExecutionProvider,
    TensorRTExecutionProvider,
};
use thiserror::Error;

pub use ort::{
    inputs,
    Session,
};

pub mod models;


pub struct BevyOrtPlugin;
impl Plugin for BevyOrtPlugin {
    fn build(&self, app: &mut App) {
        // TODO: configurable execution providers via plugin settings
        ort::init()
            .with_execution_providers([
                CoreMLExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
                OpenVINOExecutionProvider::default().build(),
                DirectMLExecutionProvider::default().build(),
                TensorRTExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])
            .commit().ok();

        app.init_asset::<Onnx>();
        app.init_asset_loader::<OnnxLoader>();
    }
}


pub enum OrtSession {
    Session(ort::Session),
    InMemory(ort::InMemorySession<'static>),
}

impl OrtSession {
    pub fn run<'s, 'i, 'v: 'i, const N: usize>(
        &'s self,
        input_values: impl Into<ort::SessionInputs<'i, 'v, N>>,
    ) -> Result<ort::SessionOutputs, ort::Error> {
        match self {
            OrtSession::Session(session) => session.run(input_values),
            OrtSession::InMemory(session) => session.run(input_values),
        }
    }

    pub fn inputs(&self) -> &Vec<ort::Input> {
        match self {
            OrtSession::Session(session) => &session.inputs,
            OrtSession::InMemory(session) => &session.inputs,
        }
    }

    pub fn outputs(&self) -> &Vec<ort::Output> {
        match self {
            OrtSession::Session(session) => &session.outputs,
            OrtSession::InMemory(session) => &session.outputs,
        }
    }
}

#[derive(Asset, Default, TypePath)]
pub struct Onnx {
    pub session_data: Vec<u8>,
    pub session: Arc<Mutex<Option<OrtSession>>>,
}

impl Onnx {
    pub fn from_session(session: Session) -> Self {
        Self {
            session_data: Vec::new(),
            session: Arc::new(Mutex::new(Some(OrtSession::Session(session)))),
        }
    }

    pub fn from_in_memory(session: ort::InMemorySession<'static>) -> Self {
        Self {
            session_data: Vec::new(),
            session: Arc::new(Mutex::new(Some(OrtSession::InMemory(session)))),
        }
    }
}


#[derive(Debug, Error)]
pub enum BevyOrtError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ort error: {0}")]
    Ort(#[from] ort::Error),
}


#[derive(Default)]
pub struct OnnxLoader;
impl AssetLoader for OnnxLoader {
    type Asset = Onnx;
    type Settings = ();
    type Error = BevyOrtError;

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a Self::Settings,
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await.map_err(BevyOrtError::from)?;

            match load_context.path().extension() {
                Some(ext) if ext == "onnx" => {
                    // TODO: add session configuration
                    let session = Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .commit_from_memory(&bytes)?;

                    Ok(Onnx::from_session(session))
                },
                _ => Err(BevyOrtError::Io(std::io::Error::new(ErrorKind::Other, "only .onnx supported"))),
            }
        })
    }

    fn extensions(&self) -> &[&str] {
        &["onnx"]
    }
}
