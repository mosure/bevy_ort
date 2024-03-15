use bevy::{
    prelude::*,
    window::PrimaryWindow,
};

use bevy_ort::{
    BevyOrtPlugin,
    inputs,
    models::yolo_v8::{
        BoundingBox,
        prepare_input,
        process_output,
    },
    Onnx,
};


fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            BevyOrtPlugin,
        ))
        .init_resource::<YoloV8>()
        .add_systems(Startup, load_yolo_v8)
        .add_systems(Update, inference)
        .run();
}


#[derive(Resource, Default)]
pub struct YoloV8 {
    pub onnx: Handle<Onnx>,
    pub input: Handle<Image>,
}


fn load_yolo_v8(
    asset_server: Res<AssetServer>,
    mut yolo_v8: ResMut<YoloV8>,
) {
    let yolo_v8_handle: Handle<Onnx> = asset_server.load("yolov8n.onnx");
    yolo_v8.onnx = yolo_v8_handle;

    let input_handle: Handle<Image> = asset_server.load("person.png");
    yolo_v8.input = input_handle;
}


fn inference(
    mut commands: Commands,
    yolo_v8: Res<YoloV8>,
    onnx_assets: Res<Assets<Onnx>>,
    images: Res<Assets<Image>>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
    mut complete: Local<bool>,
) {
    if *complete {
        return;
    }

    let window = primary_window.single();

    let image = images.get(&yolo_v8.input).expect("failed to get image asset");

    let bounding_boxes: Result<Vec<BoundingBox>, String> = (|| {
        let onnx = onnx_assets.get(&yolo_v8.onnx).ok_or("failed to get ONNX asset")?;
        let session_lock = onnx.session.lock().map_err(|e| e.to_string())?;
        let session = session_lock.as_ref().ok_or("failed to get session from ONNX asset")?;

        let model_width = session.inputs[0].input_type.tensor_dimensions().unwrap()[2] as u32;
        let model_height = session.inputs[0].input_type.tensor_dimensions().unwrap()[3] as u32;

        let input = prepare_input(image, model_width, model_height);

        let input_values = inputs!["images" => &input.as_standard_layout()].map_err(|e| e.to_string())?;
        let outputs = session.run(input_values).map_err(|e| e.to_string());

        let binding = outputs.ok().unwrap();
        let output_value: &ort::Value = binding.get("output0").unwrap();

        Ok(process_output(
            output_value,
            image.width(),
            image.height(),
            model_width,
            model_height,
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
                    image: UiImage::new(yolo_v8.input.clone()),
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
