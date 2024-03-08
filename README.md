# bevy_ort 🪨
[![test](https://github.com/mosure/bevy_ort/workflows/test/badge.svg)](https://github.com/Mosure/bevy_ort/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/bevy_ort)](https://raw.githubusercontent.com/mosure/bevy_ort/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/mosure/bevy_ort)](https://github.com/mosure/bevy_ort)
[![GitHub Releases](https://img.shields.io/github/v/release/mosure/bevy_ort?include_prereleases&sort=semver)](https://github.com/mosure/bevy_ort/releases)
[![GitHub Issues](https://img.shields.io/github/issues/mosure/bevy_ort)](https://github.com/mosure/bevy_ort/issues)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/mosure/bevy_ort.svg)](http://isitmaintained.com/project/mosure/bevy_ort)
[![crates.io](https://img.shields.io/crates/v/bevy_ort.svg)](https://crates.io/crates/bevy_ort)

a bevy plugin for the [ort](https://docs.rs/ort/latest/ort/) library


## capabilities

- [X] load ONNX models as ORT session assets
- [X] initialize ORT with default execution providers



## library usage

```rust
use bevy_ort::{
    BevyOrtPlugin,
};


fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            BevyOrtPlugin,
        ))
        .add_systems(Startup, load_model)
        .add_system(Update, inference)
        .run();
}

fn load_model(
    asset_server: Res<AssetServer>,
) {
    let model_handle: Handle<Onnx> = asset_server.load("path/to/model.onnx");
}

fn inference(
    asset_server: Res<AssetServer>,
    mut models: ResMut<Assets<Onnx>>,
) {
    let model_handle: Handle<Onnx> = todo!();

    if Some(LoadState::Loaded) == asset_server.get_load_state(model_handle) {
        let model: &Onnx = models.get(model_handle).unwrap();

        if let Some(session) = &model.session {
            let input_values = todo!();
            let outputs = session.run(input_values).unwrap();
        }
    }
}
```


## run the example person segmentation model

```sh
cargo run --bin modnet -- --input assets/person.jpg
```


## compatible bevy versions

| `bevy_ort`    | `bevy` |
| :--                   | :--    |
| `0.1.0`               | `0.13` |
