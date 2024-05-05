# bevy_ort 🪨
[![test](https://github.com/mosure/bevy_ort/workflows/test/badge.svg)](https://github.com/Mosure/bevy_ort/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/badge/license-MIT%2FGPL%E2%80%933.0-blue.svg)](https://github.com/mosure/bevy_ort#license)
[![crates.io](https://img.shields.io/crates/v/bevy_ort.svg)](https://crates.io/crates/bevy_ort)

a bevy plugin for the [ort](https://docs.rs/ort/latest/ort/) library


![person](assets/images/person.png)
![mask](assets/images/mask.png)

*> modnet inference example*


## capabilities

- [X] load ONNX models as ORT session assets
- [X] initialize ORT with default execution providers
- [X] modnet bevy image <-> ort tensor IO (with feature `modnet`)
- [X] batched modnet preprocessing
- [X] compute task pool inference scheduling

### models
- [X] lightglue (feature matching)
- [X] modnet (photographic portrait matting)
- [X] yolo_v8 (object detection)
- [X] flame (parametric head model)


## library usage

```rust
use bevy::prelude::*;

use bevy_ort::{
    BevyOrtPlugin,
    models::flame::{
        FlameInput,
        FlameOutput,
        Flame,
        FlamePlugin,
    },
};


fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            BevyOrtPlugin,
            FlamePlugin,
        ))
        .add_systems(Startup, load_flame)
        .add_systems(Startup, setup)
        .add_systems(Update, on_flame_output)
        .run();
}


fn load_flame(
    asset_server: Res<AssetServer>,
    mut flame: ResMut<Flame>,
) {
    flame.onnx = asset_server.load("models/flame.onnx");
}


fn setup(
    mut commands: Commands,
) {
    commands.spawn(FlameInput::default());
    commands.spawn(Camera3dBundle::default());
}


#[derive(Debug, Component, Reflect)]
struct HandledFlameOutput;

fn on_flame_output(
    mut commands: Commands,
    flame_outputs: Query<
        (
            Entity,
            &FlameOutput,
        ),
        Without<HandledFlameOutput>,
    >,
) {
    for (entity, flame_output) in flame_outputs.iter() {
        commands.entity(entity)
            .insert(HandledFlameOutput);

        println!("{:?}", flame_output);
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


## license

This software is dual-licensed under the MIT License and the GNU General Public License version 3 (GPL-3.0).

You may choose to use this software under the terms of the MIT License OR the GNU General Public License version 3 (GPL-3.0), except as stipulated below:

The use of the `yolo_v8` feature within this software is specifically governed by the GNU General Public License version 3 (GPL-3.0). By using the `yolo_v8` feature, you agree to comply with the terms and conditions of the GPL-3.0.

For more details on the licenses, please refer to the LICENSE.MIT and LICENSE.GPL-3.0 files included with this software.
