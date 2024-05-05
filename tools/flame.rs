use bevy::prelude::*;
use bevy_panorbit_camera::{
    PanOrbitCamera,
    PanOrbitCameraPlugin,
};

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
            PanOrbitCameraPlugin,
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
    commands.spawn((
        Camera3dBundle::default(),
        PanOrbitCamera {
            allow_upside_down: true,
            ..default()
        },
    ));
}


#[derive(Debug, Component, Reflect)]
struct HandledFlameOutput;

fn on_flame_output(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
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

            commands.spawn(PbrBundle {
                mesh: meshes.add(flame_output.mesh()),
                ..default()
            });
    }
}
