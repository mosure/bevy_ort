use bevy::prelude::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    inputs,
    Onnx,
};



pub struct FlamePlugin;
impl Plugin for FlamePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Flame>();

        app.register_type::<FlameInput>();
        app.register_type::<FlameOutput>();

        app.add_systems(PreUpdate, flame_inference_system);
    }
}

#[derive(Resource, Default)]
pub struct Flame {
    pub onnx: Handle<Onnx>,
}


fn flame_inference_system(
    mut commands: Commands,
    flame: Res<Flame>,
    onnx_assets: Res<Assets<Onnx>>,
    flame_inputs: Query<
        (
            Entity,
            &FlameInput,
        ),
        Without<FlameOutput>,
    >,
) {
    for (entity, flame_input) in flame_inputs.iter() {
        let flame_output: Result<FlameOutput, String> = (|| {
            let onnx = onnx_assets.get(&flame.onnx).ok_or("failed to get flame ONNX asset")?;
            let session_lock = onnx.session.lock().map_err(|e| e.to_string())?;
            let session = session_lock.as_ref().ok_or("failed to get flame session from flame ONNX asset")?;

            Ok(flame_inference(
                session,
                flame_input,
            ))
        })();

        match flame_output {
            Ok(flame_output) => {
                commands.entity(entity)
                    .insert(flame_output);
            }
            Err(e) => {
                warn!("{}", e);
            }
        }
    }
}


#[derive(
    Debug,
    Clone,
    Component,
    Reflect,
)]
pub struct FlameInput {
    pub shape: [[f32; 100]; 8],
    pub pose: [[f32; 6]; 8],
    pub expression: [[f32; 50]; 8],
    pub neck: [[f32; 3]; 8],
    pub eye: [[f32; 6]; 8],
}

impl Default for FlameInput {
    fn default() -> Self {
        let radian = std::f32::consts::PI / 180.0;

        Self {
            shape: [[0.0; 100]; 8],
            pose: [
                [0.0, 30.0 * radian, 0.0, 0.0, 0.0, 0.0],
                [0.0, -30.0 * radian, 0.0, 0.0, 0.0, 0.0],
                [0.0, 85.0 * radian, 0.0, 0.0, 0.0, 0.0],
                [0.0, -48.0 * radian, 0.0, 0.0, 0.0, 0.0],
                [0.0, 10.0 * radian, 0.0, 0.0, 0.0, 0.0],
                [0.0, -15.0 * radian, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0 * radian, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0 * radian, 0.0, 0.0, 0.0, 0.0],
            ],
            expression: [[0.0; 50]; 8],
            neck: [[0.0; 3]; 8],
            eye: [[0.0; 6]; 8],
        }
    }
}


#[derive(
    Debug,
    Default,
    Clone,
    Component,
    Deserialize,
    Serialize,
    Reflect,
)]
pub struct FlameOutput {
    pub vertices: Vec<[f32; 3]>,  // TODO: use Vec3 for binding
    pub landmarks: Vec<[f32; 3]>,
}


pub fn flame_inference(
    session: &ort::Session,
    input: &FlameInput,
) -> FlameOutput {
    let PreparedInput {
        shape,
        expression,
        pose,
        neck,
        eye,
     } = prepare_input(input);

    let input_values = inputs![
        "shape" => shape.view(),
        "expression" => expression.view(),
        "pose" => pose.view(),
        "neck" => neck.view(),
        "eye" => eye.view(),
    ].map_err(|e| e.to_string()).unwrap();
    let outputs = session.run(input_values).map_err(|e| e.to_string());
    let binding = outputs.ok().unwrap();

    let vertices: &ort::Value = binding.get("vertices").unwrap();
    let landmarks: &ort::Value = binding.get("landmarks").unwrap();

    post_process(
        vertices,
        landmarks,
    )
}


pub struct PreparedInput {
    pub shape: Array2<f32>,
    pub pose: Array2<f32>,
    pub expression: Array2<f32>,
    pub neck: Array2<f32>,
    pub eye: Array2<f32>,
}

pub fn prepare_input(
    input: &FlameInput,
) -> PreparedInput {
    let shape = Array2::from_shape_vec((8, 100), input.shape.concat()).unwrap();
    let pose = Array2::from_shape_vec((8, 6), input.pose.concat()).unwrap();
    let expression = Array2::from_shape_vec((8, 50), input.expression.concat()).unwrap();
    let neck = Array2::from_shape_vec((8, 3), input.neck.concat()).unwrap();
    let eye = Array2::from_shape_vec((8, 6), input.eye.concat()).unwrap();

    PreparedInput {
        shape,
        expression,
        pose,
        neck,
        eye,
    }
}


pub fn post_process(
    vertices: &ort::Value,
    landmarks: &ort::Value,
) -> FlameOutput {
    let vertices_tensor = vertices.try_extract_tensor::<f32>().unwrap();
    let vertices_view = vertices_tensor.view();  // [8, 5023, 3]

    let landmarks_tensor = landmarks.try_extract_tensor::<f32>().unwrap();
    let landmarks_view = landmarks_tensor.view();  // [8, 68, 3]

    let vertices = vertices_view.outer_iter()
        .flat_map(|subtensor| {
            subtensor.outer_iter().map(|row| {
                [row[0], row[1], row[2]]
            }).collect::<Vec<[f32; 3]>>()
        })
        .collect::<Vec::<_>>();

    let landmarks = landmarks_view.outer_iter()
        .flat_map(|subtensor| {
            subtensor.outer_iter().map(|row| {
                [row[0], row[1], row[2]]
            }).collect::<Vec<[f32; 3]>>()
        })
        .collect::<Vec::<_>>();

    FlameOutput {
        vertices,
        landmarks,
    }
}
