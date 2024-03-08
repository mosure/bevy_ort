use std::io::ErrorKind;

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
    CUDAExecutionProvider,
    GraphOptimizationLevel,
    OpenVINOExecutionProvider,
};
use thiserror::Error;

pub use ort::{
    inputs,
    Session,
};


pub struct BevyOrtPlugin;
impl Plugin for BevyOrtPlugin {
    fn build(&self, app: &mut App) {
        // TODO: configurable execution providers via plugin settings
        ort::init()
            .with_execution_providers([
                CoreMLExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
                OpenVINOExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])
            .commit().ok();

        app.init_asset::<Onnx>();
        app.init_asset_loader::<OnnxLoader>();
    }
}


#[derive(Asset, Debug, Default, TypePath)]
pub struct Onnx {
    pub session: Option<Session>,
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
                        .with_intra_threads(4)?
                        .with_model_from_memory(&bytes)?;

                    Ok(Onnx {
                        session: Some(session),
                    })
                },
                _ => Err(BevyOrtError::Io(std::io::Error::new(ErrorKind::Other, "only .onnx supported"))),
            }
        })
    }

    fn extensions(&self) -> &[&str] {
        &["onnx"]
    }
}
