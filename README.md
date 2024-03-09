# bevy_ort 🪨
[![test](https://github.com/mosure/bevy_ort/workflows/test/badge.svg)](https://github.com/Mosure/bevy_ort/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/bevy_ort)](https://raw.githubusercontent.com/mosure/bevy_ort/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/mosure/bevy_ort)](https://github.com/mosure/bevy_ort)
[![GitHub Releases](https://img.shields.io/github/v/release/mosure/bevy_ort?include_prereleases&sort=semver)](https://github.com/mosure/bevy_ort/releases)
[![GitHub Issues](https://img.shields.io/github/issues/mosure/bevy_ort)](https://github.com/mosure/bevy_ort/issues)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/mosure/bevy_ort.svg)](http://isitmaintained.com/project/mosure/bevy_ort)
[![crates.io](https://img.shields.io/crates/v/bevy_ort.svg)](https://crates.io/crates/bevy_ort)

a bevy plugin for the [ort](https://docs.rs/ort/latest/ort/) library


![person](assets/person.png)
![mask](assets/mask.png)

*> modnet inference example*


## capabilities

- [X] load ONNX models as ORT session assets
- [X] initialize ORT with default execution providers
- [X] modnet bevy image <-> ort tensor IO (with feature `modnet`)
- [ ] compute task pool inference scheduling


## library usage

```rust
use bevy::prelude::*;

use bevy_ort::{
    BevyOrtPlugin,
    inputs,
    models::modnet::{
        image_to_modnet_input,
        modnet_output_to_luma_image,
    },
    Onnx,
};


fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            BevyOrtPlugin,
        ))
        .init_resource::<Modnet>()
        .add_systems(Startup, load_modnet)
        .add_systems(Update, inference)
        .run();
}

#[derive(Resource, Default)]
pub struct Modnet {
    pub onnx: Handle<Onnx>,
    pub input: Handle<Image>,
}

fn load_modnet(
    asset_server: Res<AssetServer>,
    mut modnet: ResMut<Modnet>,
) {
    let modnet_handle: Handle<Onnx> = asset_server.load("modnet_photographic_portrait_matting.onnx");
    modnet.onnx = modnet_handle;

    let input_handle: Handle<Image> = asset_server.load("person.png");
    modnet.input = input_handle;
}


fn inference(
    mut commands: Commands,
    modnet: Res<Modnet>,
    onnx_assets: Res<Assets<Onnx>>,
    mut images: ResMut<Assets<Image>>,
    mut complete: Local<bool>,
) {
    if *complete {
        return;
    }

    let image = images.get(&modnet.input).expect("failed to get image asset");
    let input = image_to_modnet_input(image);

    let output: Result<ort::SessionOutputs<'_>, String> = (|| {
        let onnx = onnx_assets.get(&modnet.onnx).ok_or("failed to get ONNX asset")?;
        let session = onnx.session.as_ref().ok_or("failed to get session from ONNX asset")?;

        let input_values = inputs!["input" => input.view()].map_err(|e| e.to_string())?;
        session.run(input_values).map_err(|e| e.to_string())
    })();

    match output {
        Ok(output) => {
            let output_value: &ort::Value = output.get("output").unwrap();

            let mask_image = modnet_output_to_luma_image(output_value);
            let mask_image = images.add(mask_image);

            commands.spawn(NodeBundle {
                style: Style {
                    display: Display::Grid,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    grid_template_columns: RepeatedGridTrack::flex(1, 1.0),
                    grid_template_rows: RepeatedGridTrack::flex(1, 1.0),
                    ..default()
                },
                background_color: BackgroundColor(Color::DARK_GRAY),
                ..default()
            })
            .with_children(|builder| {
                builder.spawn(ImageBundle {
                    style: Style {
                        ..default()
                    },
                    image: UiImage::new(mask_image.clone()),
                    ..default()
                });
            });

            commands.spawn(Camera2dBundle::default());

            *complete = true;
        },
        Err(error) => {
            println!("inference failed: {}", error);
        }
    }
}

```


## run the example person segmentation model (modnet)

```sh
cargo run
```

use an accelerated execution provider:
- windows - `cargo run --features ort/cuda` or `cargo run --features ort/openvino`
- macos - `cargo run --features ort/coreml`
- linux - `cargo run --features ort/tensorrt` or `cargo run --features ort/openvino`

> see complete list of ort features here: https://github.com/pykeio/ort/blob/0aec4030a5f3470e4ee6c6f4e7e52d4e495ec27a/Cargo.toml#L54

> note: if you use `pip install onnxruntime`, you may need to run `ORT_STRATEGY=system cargo run`, see: https://docs.rs/ort/latest/ort/#how-to-get-binaries


## compatible bevy versions

| `bevy_ort`    | `bevy` |
| :--                   | :--    |
| `0.1.0`               | `0.13` |

## credits
- [modnet](https://github.com/ZHKKKe/MODNet)
