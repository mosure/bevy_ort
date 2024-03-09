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
            eprintln!("inference failed: {}", error);
        }
    }
}
