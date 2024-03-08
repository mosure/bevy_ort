use bevy::{
    prelude::*,
    app::AppExit,
};
use ndarray::Array;

use bevy_ort::{
    BevyOrtPlugin,
    inputs,
    Onnx,
};


// TODO: provide an example /w AsyncComputeTask inference
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
}

fn load_modnet(
    asset_server: Res<AssetServer>,
    mut modnet: ResMut<Modnet>,
) {
    let modnet_handle: Handle<Onnx> = asset_server.load("modnet_photographic_portrait_matting.onnx");
    modnet.onnx = modnet_handle;
}


fn inference(
    mut exit: EventWriter<AppExit>,
    modnet: Res<Modnet>,
    onnx_assets: Res<Assets<Onnx>>,
) {
    let input = Array::<f32, _>::zeros((1, 3, 640, 640));

    let output = (|| -> Option<_> {
        let onnx = onnx_assets.get(&modnet.onnx)?;
        let session = onnx.session.as_ref()?;

        println!("inputs: {:?}", session.inputs);

        let input_values = inputs!["image" => input.view()].unwrap();
        session.run(input_values).ok()
    })();

    if let Some(output) = output {
        println!("outputs: {:?}", output.keys());

        exit.send(AppExit);
    } else {
        // TODO: use Result instead of Option for error reporting
        println!("inference failed");
    }
}
