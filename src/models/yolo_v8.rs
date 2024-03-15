use bevy::prelude::*;
use image::GenericImageView;
use ndarray::{Array, ArrayD, Axis};


#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub prob: f32,
}


pub fn prepare_input(
    image: &Image,
    model_width: u32,
    model_height: u32,
) -> ArrayD<f32> {
    let image = &image.clone().try_into_dynamic().unwrap();
    let image = image.resize_exact(model_width, model_height, image::imageops::FilterType::CatmullRom);

    let mut input = Array::zeros((1, 3, model_width as usize, model_height as usize)).into_dyn();

    image.pixels().for_each(|(x, y, pixel)| {
        let [r, g, b, _] = pixel.0;
        let (x, y) = (x as usize, y as usize);

        input[[0, 0, y, x]] = r as f32 / 255.0;
        input[[0, 1, y, x]] = g as f32 / 255.0;
        input[[0, 2, y, x]] = b as f32 / 255.0;
    });

    input
}


pub fn process_output(
    output: &ort::Value,
    width: u32,
    height: u32,
    model_width: u32,
    model_height: u32,
) -> Vec<BoundingBox> {
    let mut boxes = Vec::new();

    let tensor = output.extract_tensor::<f32>().unwrap();
    let data = tensor.view().t().into_owned();

    for detection  in data.axis_iter(Axis(0)) {
        let detection : Vec<_> = detection.iter().collect();

        let (class_id, prob) = detection.iter()
            .skip(4)
            .enumerate()
            .reduce(|acc, row| if row.1 > acc.1 { row } else { acc })
            .unwrap();

        if **prob < 0.5 {
            continue;
        }

        let xc = detection[0] / model_width as f32 * width as f32;
        let yc = detection[1] / model_height as f32 * height as f32;
        let w = detection[2] / model_width as f32 * width as f32;
        let h = detection[3] / model_height as f32 * height as f32;

        let x1 = (xc - w / 2.0).max(0.0);
        let y1 = (yc - h / 2.0).max(0.0);
        let x2 = (xc + w / 2.0).min(width as f32);
        let y2 = (yc + h / 2.0).min(height as f32);

        boxes.push(BoundingBox {
            x1,
            y1,
            x2,
            y2,
            class_id,
            prob: **prob,
        });
    }

    boxes
}


pub const YOLO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];
