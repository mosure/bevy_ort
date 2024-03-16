use bevy::{
    prelude::*,
    window::PrimaryWindow,
};

use bevy_ort::{
    BevyOrtPlugin,
    models::yolo_v8::{
        yolo_inference,
        BoundingBox,
        Yolo,
        YoloPlugin,
    },
    Onnx,
};


fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            BevyOrtPlugin,
            YoloPlugin,
        ))
        .init_resource::<YoloInput>()
        .add_systems(Startup, load_yolo)
        .add_systems(Update, inference)
        .run();
}


#[derive(Resource, Default)]
pub struct YoloInput {
    pub image: Handle<Image>,
}


fn load_yolo(
    asset_server: Res<AssetServer>,
    mut yolo: ResMut<Yolo>,
    mut input: ResMut<YoloInput>,
) {
    let yolo_v8_handle: Handle<Onnx> = asset_server.load("yolov8n.onnx");
    yolo.onnx = yolo_v8_handle;

    let input_handle: Handle<Image> = asset_server.load("person.png");
    input.image = input_handle;
}


fn inference(
    mut commands: Commands,
    yolo: Res<Yolo>,
    input: Res<YoloInput>,
    onnx_assets: Res<Assets<Onnx>>,
    images: Res<Assets<Image>>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
    mut complete: Local<bool>,
) {
    if *complete {
        return;
    }

    let window = primary_window.single();

    let image = images.get(&input.image).expect("failed to get image asset");

    let bounding_boxes: Result<Vec<BoundingBox>, String> = (|| {
        let onnx = onnx_assets.get(&yolo.onnx).ok_or("failed to get ONNX asset")?;
        let session_lock = onnx.session.lock().map_err(|e| e.to_string())?;
        let session = session_lock.as_ref().ok_or("failed to get session from ONNX asset")?;

        Ok(yolo_inference(
            session,
            image,
            0.5,
        ))
    })();

    match bounding_boxes {
        Ok(bounding_boxes) => {

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
                    image: UiImage::new(input.image.clone()),
                    ..default()
                });

                if let Some(first_box) = bounding_boxes.first() {
                    let x1 = first_box.x1 / image.width() as f32 * window.width();
                    let y1 = first_box.y1 / image.height() as f32 * window.height();

                    let bb_width = (first_box.x2 - first_box.x1) / image.width() as f32 * window.width();
                    let bb_height = (first_box.y2 - first_box.y1) / image.height() as f32 * window.height();

                    builder.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            left: Val::Px(x1),
                            top: Val::Px(y1),
                            width: Val::Px(bb_width),
                            height: Val::Px(bb_height),
                            border: UiRect::all(Val::Px(2.0)),
                            ..default()
                        },
                        background_color: BackgroundColor(Color::rgba(1.0, 0.0, 0.0, 0.5)),
                        ..default()
                    });
                }
            });

            commands.spawn(Camera2dBundle::default());

            *complete = true;
        }
        Err(e) => {
            eprintln!("inference failed: {}", e);
        }
    }
}
