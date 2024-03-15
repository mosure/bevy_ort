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
    models::modnet::{
        modnet_output_to_luma_images,
        images_to_modnet_input,
    },
    Session,
};
use ort::GraphOptimizationLevel;


const MAX_RESOLUTIONS: [(u32, u32); 4] = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
];

const STREAM_COUNT: usize = 16;


criterion_group!{
    name = modnet_benches;
    config = Criterion::default().sample_size(10);
    targets = images_to_modnet_input_benchmark,
              modnet_output_to_luma_images_benchmark,
              modnet_inference_benchmark,
}
criterion_main!(modnet_benches);


fn images_to_modnet_input_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("images_to_modnet_input");

    MAX_RESOLUTIONS.iter()
        .for_each(|(width, height)| {
            let data = vec![0u8; (1920 * 1080 * 4) as usize];

            let images = (0..STREAM_COUNT)
                .map(|_|{
                    Image::new(
                        Extent3d {
                            width: 1920,
                            height: 1080,
                            depth_or_array_layers: 1,
                        },
                        TextureDimension::D2,
                        data.clone(),
                        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                        RenderAssetUsages::all(),
                    )
                })
                .collect::<Vec<_>>();

            group.throughput(Throughput::Elements(STREAM_COUNT as u64));
            group.bench_with_input(BenchmarkId::from_parameter(format!("{}x{}", width, height)), &images, |b, images| {
                let views = images.iter().map(|image| image).collect::<Vec<_>>();

                b.iter(|| images_to_modnet_input(views.as_slice(), Some((*width, *height))));
            });
        });
}


fn modnet_output_to_luma_images_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("modnet_output_to_luma_images");

    let session = Session::builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .with_model_from_file("assets/modnet_photographic_portrait_matting.onnx").unwrap();

    let data = vec![0u8; (1920 * 1080 * 4) as usize];
    let image: Image = Image::new(
        Extent3d {
            width: 1920,
            height: 1080,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data.clone(),
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    );

    MAX_RESOLUTIONS.iter()
        .for_each(|size_limit| {
            let input = images_to_modnet_input(&[&image; STREAM_COUNT], size_limit.clone().into());
            let input_values = inputs!["input" => input.view()].map_err(|e| e.to_string()).unwrap();

            let outputs = session.run(input_values).map_err(|e| e.to_string());
            let binding = outputs.ok().unwrap();
            let output_value: &ort::Value = binding.get("output").unwrap();

            group.throughput(Throughput::Elements(STREAM_COUNT as u64));
            group.bench_with_input(BenchmarkId::from_parameter(format!("{}x{}", size_limit.0, size_limit.1)), &output_value, |b, output_value| {
                b.iter(|| modnet_output_to_luma_images(output_value));
            });
        });
}


fn modnet_inference_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("modnet_inference");

    let session = Session::builder().unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .with_model_from_file("assets/modnet_photographic_portrait_matting.onnx").unwrap();

    MAX_RESOLUTIONS.iter().for_each(|(width, height)| {
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

        let input = images_to_modnet_input(&[&image], Some((*width, *height)));

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{}x{}", width, height)), &(width, height), |b, _| {
            b.iter(|| {
                let input_values = inputs!["input" => input.view()].map_err(|e| e.to_string()).unwrap();
                let outputs = session.run(input_values).map_err(|e| e.to_string());
                let binding = outputs.ok().unwrap();
                let output_value: &ort::Value = binding.get("output").unwrap();
                modnet_output_to_luma_images(output_value);
            });
        });
    });
}
