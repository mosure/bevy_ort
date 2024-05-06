use criterion::{
    BenchmarkId,
    criterion_group,
    criterion_main,
    Criterion,
    Throughput,
};

use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{
            Extent3d,
            TextureDimension,
        },
    },
};
use bevy_ort::{
    inputs,
    models::yolo_v8::{
        prepare_input,
        process_output,
        yolo_inference,
    },
    OrtSession,
    Session,
};
use ort::GraphOptimizationLevel;


criterion_group!{
    name = yolo_v8_benches;
    config = Criterion::default().sample_size(10);
    targets = prepare_input_benchmark,
        process_output_benchmark,
        inference_benchmark,
}
criterion_main!(yolo_v8_benches);


const RESOLUTIONS: [(u32, u32); 3] = [
    (640, 360),
    (1280, 720),
    (1920, 1080),
];

// TODO: read input shape from session
const MODEL_WIDTH: u32 = 640;
const MODEL_HEIGHT: u32 = 640;


fn prepare_input_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("yolo_v8_prepare_input");

    RESOLUTIONS.iter()
        .for_each(|(width, height)| {
            let data = vec![0u8; (width * height * 4) as usize];
            let image = Image::new(
                Extent3d {
                    width: *width,
                    height: *height,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                data.clone(),
                bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::all(),
            );

            group.throughput(Throughput::Elements(1));
            group.bench_with_input(BenchmarkId::from_parameter(format!("{}x{}", width, height)), &image, |b, image| {
                b.iter(|| prepare_input(&image, MODEL_WIDTH, MODEL_HEIGHT));
            });
        });
}


fn process_output_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("yolo_v8_process_output");

    let session = Session::builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .commit_from_file("assets/yolov8n.onnx").unwrap();

    RESOLUTIONS.iter()
        .for_each(|(width, height)| {
            let data = vec![0u8; (width * height * 4) as usize];
            let image: Image = Image::new(
                Extent3d {
                    width: *width,
                    height: *height,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                data.clone(),
                bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::all(),
            );

            let input = prepare_input(&image, MODEL_WIDTH, MODEL_HEIGHT);
            let input_values = inputs!["images" => &input.as_standard_layout()].map_err(|e| e.to_string()).unwrap();

            let outputs = session.run(input_values).map_err(|e| e.to_string());
            let binding = outputs.ok().unwrap();
            let output_value: &ort::Value = binding.get("output0").unwrap();

            group.throughput(Throughput::Elements(1));
            group.bench_with_input(BenchmarkId::from_parameter(format!("{}x{}", width, height)), &output_value, |b, output_value| {
                b.iter(|| process_output(output_value, *width, *height, MODEL_WIDTH, MODEL_HEIGHT));
            });
        });
}


fn inference_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("yolo_v8_inference");

    let session = Session::builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .commit_from_file("assets/yolov8n.onnx").unwrap();
    let session = OrtSession::Session(session);

    RESOLUTIONS.iter().for_each(|(width, height)| {
        let data = vec![0u8; *width as usize * *height as usize * 4];
        let image = Image::new(
            Extent3d {
                width: *width,
                height: *height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data.clone(),
            bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::all(),
        );

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{}x{}", width, height)), &(width, height), |b, _| {
            b.iter(|| {
                yolo_inference(&session, &image, 0.5)
            });
        });
    });
}
