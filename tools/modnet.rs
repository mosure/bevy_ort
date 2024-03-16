use bevy::prelude::*;

use bevy_ort::{
    BevyOrtPlugin,
    models::modnet::{
        modnet_inference,
        Modnet,
        ModnetPlugin,
    },
    Onnx,
};


// see async inference example in bevy_light_field: https://github.com/mosure/bevy_light_field/blob/ba8a5c09eebb68d820676fa18cdb56a621fbdcb8/src/matting.rs#L58-L59
fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            BevyOrtPlugin,
            ModnetPlugin,
        ))
        .init_resource::<ModnetInput>()
        .add_systems(Startup, load_modnet)
        .add_systems(Update, inference)
        .run();
}

#[derive(Resource, Default)]
struct ModnetInput {
    image: Handle<Image>,
}


fn load_modnet(
    asset_server: Res<AssetServer>,
    mut modnet: ResMut<Modnet>,
    mut input: ResMut<ModnetInput>,
) {
    let modnet_handle: Handle<Onnx> = asset_server.load("modnet_photographic_portrait_matting.onnx");
    modnet.onnx = modnet_handle;

    let input_handle: Handle<Image> = asset_server.load("person.png");
    input.image = input_handle;
}


fn inference(
    mut commands: Commands,
    modnet: Res<Modnet>,
    input: Res<ModnetInput>,
    onnx_assets: Res<Assets<Onnx>>,
    mut images: ResMut<Assets<Image>>,
    mut complete: Local<bool>,
) {
    if *complete {
        return;
    }

    let image = images.get(&input.image).expect("failed to get image asset");

    let mask_image: Result<Image, String> = (|| {
        let onnx = onnx_assets.get(&modnet.onnx).ok_or("failed to get ONNX asset")?;
        let session_lock = onnx.session.lock().map_err(|e| e.to_string())?;
        let session = session_lock.as_ref().ok_or("failed to get session from ONNX asset")?;

        Ok(modnet_inference(session, &[image], None).pop().unwrap())
    })();

    match mask_image {
        Ok(mask_image) => {
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
        }
        Err(e) => {
            eprintln!("inference failed: {}", e);
        }
    }
}
