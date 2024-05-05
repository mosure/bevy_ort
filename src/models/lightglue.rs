use bevy::prelude::*;
use image::GenericImageView;
use ndarray::{Array, ArrayD, Axis};
use serde::{Deserialize, Serialize};

use crate::{
    inputs,
    Onnx,
};



pub struct LightgluePlugin;
impl Plugin for LightgluePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Lightglue>();
    }
}

#[derive(Resource, Default)]
pub struct Lightglue {
    pub onnx: Handle<Onnx>,
}


#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct GluedPair {
    pub from_x: i64,
    pub from_y: i64,
    pub to_x: i64,
    pub to_y: i64,
}


pub fn lightglue_inference(
    session: &ort::Session,
    images: &[&Image],
) -> Vec<(usize, usize, Vec<GluedPair>)> {
    let unique_unordered_pairs = images.iter().enumerate()
        .flat_map(|(i, _)| {
            images.iter().enumerate().skip(i + 1).map(move |(j, _)| (i, j))
        })
        .collect::<Vec<_>>();

    unique_unordered_pairs.iter()
        .map(|(i, j)| {
            let a = images[*i];
            let b = images[*j];

            let prepared_a = prepare_input(a);
            let prepared_b = prepare_input(b);

            let input_values = inputs![
                "image0" => prepared_a.view(),
                "image1" => prepared_b.view(),
            ].map_err(|e| e.to_string()).unwrap();
            let outputs = session.run(input_values).map_err(|e| e.to_string());
            let binding = outputs.ok().unwrap();

            let kpts0: &ort::Value = binding.get("kpts0").unwrap();
            let kpts1: &ort::Value = binding.get("kpts1").unwrap();
            let matches0: &ort::Value = binding.get("matches0").unwrap();

            (
                *i,
                *j,
                post_process(
                    kpts0,
                    kpts1,
                    matches0,
                ).unwrap(),
            )
        })
        .collect::<Vec<_>>()
}


pub fn prepare_input(
    image: &Image,
) -> ArrayD<f32> {
    let image = &image.clone().try_into_dynamic().unwrap();

    let mut input = Array::zeros((1, 3, image.height() as usize, image.width() as usize)).into_dyn();

    image.pixels().for_each(|(x, y, pixel)| {
        let [r, g, b, _] = pixel.0;
        let (x, y) = (x as usize, y as usize);

        input[[0, 0, y, x]] = r as f32 / 255.0;
        input[[0, 1, y, x]] = g as f32 / 255.0;
        input[[0, 2, y, x]] = b as f32 / 255.0;
    });

    input
}


pub fn post_process(
    kpts0: &ort::Value,
    kpts1: &ort::Value,
    matches: &ort::Value,
) -> Result<Vec<GluedPair>, &'static str> {
    let kpts0_tensor = kpts0.try_extract_tensor::<i64>().unwrap();
    let kpts0_view = kpts0_tensor.view();

    let kpts1_tensor = kpts1.try_extract_tensor::<i64>().unwrap();
    let kpts1_view = kpts1_tensor.view();

    let matches = matches.try_extract_tensor::<i64>().unwrap();
    let matches_view = matches.view();

    Ok(
        matches_view.axis_iter(Axis(0))
            .map(|row| {
                let kpts0_idx = row[0];
                let kpts1_idx = row[1];

                let kpt0 = kpts0_view.index_axis(Axis(1), kpts0_idx as usize);

                let kpt0_x = kpt0[[0, 0]];
                let kpt0_y = kpt0[[0, 1]];

                let kpt1 = kpts1_view.index_axis(Axis(1), kpts1_idx as usize);
                let kpt1_x = kpt1[[0, 0]];
                let kpt1_y = kpt1[[0, 1]];

                GluedPair {
                    from_x: kpt0_x,
                    from_y: kpt0_y,
                    to_x: kpt1_x,
                    to_y: kpt1_y,
                }
            })
            .collect::<Vec<_>>()
    )
}
