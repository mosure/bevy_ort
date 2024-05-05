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
