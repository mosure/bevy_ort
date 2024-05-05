use bevy::prelude::*;

use bevy_ort::{
    BevyOrtPlugin,
    models::flame::{
        FlameInput,
        FlameOutput,
        flame_inference,
        Flame,
        FlamePlugin,
    },
    Onnx,
};


fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            BevyOrtPlugin,
            FlamePlugin,
        ))
        .add_systems(Startup, load_flame)
        .add_systems(Update, inference)
        .run();
}


fn load_flame(
    asset_server: Res<AssetServer>,
    mut flame: ResMut<Flame>,
) {
    let flame_handle: Handle<Onnx> = asset_server.load("models/flame.onnx");
    flame.onnx = flame_handle;
}


fn inference(
    mut commands: Commands,
    flame: Res<Flame>,
    onnx_assets: Res<Assets<Onnx>>,
    mut complete: Local<bool>,
) {
    if *complete {
        return;
    }

    let flame_output: Result<FlameOutput, String> = (|| {
        let onnx = onnx_assets.get(&flame.onnx).ok_or("failed to get ONNX asset")?;
        let session_lock = onnx.session.lock().map_err(|e| e.to_string())?;
        let session = session_lock.as_ref().ok_or("failed to get session from ONNX asset")?;

        Ok(flame_inference(
            session,
            &FlameInput::default(),
        ))
    })();

    match flame_output {
        Ok(_flame_output) => {
            // TODO: insert mesh
            // TODO: insert pan orbit camera
            commands.spawn(Camera3dBundle::default());
            *complete = true;
        }
        Err(e) => {
            eprintln!("inference failed: {}", e);
        }
    }
}
