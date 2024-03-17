use bevy::{
    prelude::*,
    window::PrimaryWindow,
};

use bevy_ort::{
    BevyOrtPlugin,
    models::lightglue::{
        GluedPair,
        lightglue_inference,
        Lightglue,
        LightgluePlugin,
    },
    Onnx,
};


fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            BevyOrtPlugin,
            LightgluePlugin,
        ))
        .init_resource::<LightglueInput>()
        .add_systems(Startup, load_lightglue)
        .add_systems(Update, inference)
        .run();
}


#[derive(Resource, Default)]
pub struct LightglueInput {
    pub a: Handle<Image>,
    pub b: Handle<Image>,
}


fn load_lightglue(
    asset_server: Res<AssetServer>,
    mut lightglue: ResMut<Lightglue>,
    mut input: ResMut<LightglueInput>,
) {
    let lightglue_handle: Handle<Onnx> = asset_server.load("models/disk_lightglue_end2end_fused_cpu.onnx");
    lightglue.onnx = lightglue_handle;

    input.a = asset_server.load("images/sacre_coeur1.png");
    input.b = asset_server.load("images/sacre_coeur2.png");
}


fn inference(
    mut commands: Commands,
    lightglue: Res<Lightglue>,
    input: Res<LightglueInput>,
    onnx_assets: Res<Assets<Onnx>>,
    images: Res<Assets<Image>>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
    mut complete: Local<bool>,
) {
    if *complete {
        return;
    }

    let window = primary_window.single();

    let images = [
        images.get(&input.a).expect("failed to get image asset"),
        images.get(&input.b).expect("failed to get image asset"),
    ];

    let glued_pairs: Result<Vec<(usize, usize, Vec<GluedPair>)>, String> = (|| {
        let onnx = onnx_assets.get(&lightglue.onnx).ok_or("failed to get ONNX asset")?;
        let session_lock = onnx.session.lock().map_err(|e| e.to_string())?;
        let session = session_lock.as_ref().ok_or("failed to get session from ONNX asset")?;

        Ok(lightglue_inference(
            session,
            &images,
        ))
    })();

    match glued_pairs {
        Ok(glued_pairs) => {
            println!("glued_pairs: {:?}", glued_pairs[0].2.len());

            commands.spawn(NodeBundle {
                style: Style {
                    display: Display::Grid,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    grid_template_columns: RepeatedGridTrack::flex(2, 1.0),
                    grid_template_rows: RepeatedGridTrack::flex(2, 1.0),
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
                    image: UiImage::new(input.a.clone()),
                    ..default()
                })
                .with_children(|builder| {
                    let image_width = images[0].width() as f32;
                    let image_height = images[0].height() as f32;

                    let display_width = window.width() / 2.0;
                    let display_height = window.height() / 2.0;

                    let scale_x = display_width / image_width;
                    let scale_y = display_height / image_height;

                    glued_pairs[0].2.iter().for_each(|pair| {
                        let scaled_x = pair.from_x as f32 * scale_x;
                        let scaled_y = pair.from_y as f32 * scale_y;

                        builder.spawn(NodeBundle {
                            style: Style {
                                position_type: PositionType::Absolute,
                                left: Val::Px(scaled_x),
                                top: Val::Px(scaled_y),
                                width: Val::Px(2.0),
                                height: Val::Px(2.0),
                                ..default()
                            },
                            background_color: Color::rgb(1.0, 0.0, 0.0).into(),
                            ..default()
                        });
                    });
                });

                builder.spawn(ImageBundle {
                    style: Style {
                        ..default()
                    },
                    image: UiImage::new(input.b.clone()),
                    ..default()
                })
                .with_children(|builder| {
                    let image_width = images[1].width() as f32;
                    let image_height = images[1].height() as f32;

                    let display_width = window.width() / 2.0;
                    let display_height = window.height() / 2.0;

                    let scale_x = display_width / image_width;
                    let scale_y = display_height / image_height;

                    glued_pairs[0].2.iter().for_each(|pair| {
                        let scaled_x = pair.to_x as f32 * scale_x;
                        let scaled_y = pair.to_y as f32 * scale_y;

                        builder.spawn(NodeBundle {
                            style: Style {
                                position_type: PositionType::Absolute,
                                left: Val::Px(scaled_x),
                                top: Val::Px(scaled_y),
                                width: Val::Px(2.0),
                                height: Val::Px(2.0),
                                ..default()
                            },
                            background_color: Color::rgb(0.0, 1.0, 0.0).into(),
                            ..default()
                        });
                    });
                });

                builder.spawn(ImageBundle {
                    style: Style {
                        ..default()
                    },
                    image: UiImage::new(input.a.clone()),
                    ..default()
                });

                builder.spawn(ImageBundle {
                    style: Style {
                        ..default()
                    },
                    image: UiImage::new(input.b.clone()),
                    ..default()
                });

                // TODO: draw lines between keypoints
            });

            commands.spawn(Camera2dBundle::default());

            *complete = true;
        }
        Err(e) => {
            eprintln!("inference failed: {}", e);
        }
    }
}
