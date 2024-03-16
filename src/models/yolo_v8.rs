use bevy::prelude::*;
use image::GenericImageView;
use ndarray::{Array, ArrayD, Axis};
use serde::{Deserialize, Serialize};

use crate::{
    inputs,
    Onnx,
};


#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_id: usize,
    pub prob: f32,
}


pub struct YoloPlugin;
impl Plugin for YoloPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Yolo>();
    }
}

#[derive(Resource, Default)]
pub struct Yolo {
    pub onnx: Handle<Onnx>,
}


// TODO: support yolo input batching
pub fn yolo_inference(
    session: &ort::Session,
    image: &Image,
    iou_threshold: f32,
) -> Vec<BoundingBox> {
    let width = image.width();
    let height = image.height();

    let model_width = session.inputs[0].input_type.tensor_dimensions().unwrap()[2] as u32;
    let model_height = session.inputs[0].input_type.tensor_dimensions().unwrap()[3] as u32;

    let input = prepare_input(image, model_width, model_height);

    let input_values = inputs!["images" => &input.as_standard_layout()].map_err(|e| e.to_string()).unwrap();
    let outputs = session.run(input_values).map_err(|e| e.to_string());
    let binding = outputs.ok().unwrap();
    let output_value: &ort::Value = binding.get("output0").unwrap();

    let detections = process_output(output_value, width, height, model_width, model_height);

    nms(&detections, iou_threshold)
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


pub fn nms(
    input: &[BoundingBox],
    iou_threshold: f32,
) -> Vec<BoundingBox> {
    let mut output: Vec<BoundingBox> = Vec::new();

    let mut boxes_by_class = std::collections::HashMap::new();
    for bbox in input {
        boxes_by_class.entry(bbox.class_id)
            .or_insert_with(Vec::new)
            .push(bbox.clone());
    }

    for (_class_id, mut boxes) in boxes_by_class {
        boxes.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal));

        while !boxes.is_empty() {
            let highest = boxes.remove(0);
            output.push(highest.clone());

            boxes.retain(|bbox| iou(&highest, bbox) < iou_threshold);
        }
    }

    output
}

fn iou(box_a: &BoundingBox, box_b: &BoundingBox) -> f32 {
    let intersection_x1 = box_a.x1.max(box_b.x1);
    let intersection_y1 = box_a.y1.max(box_b.y1);
    let intersection_x2 = box_a.x2.min(box_b.x2);
    let intersection_y2 = box_a.y2.min(box_b.y2);

    let intersection_area = 0f32.max(intersection_x2 - intersection_x1) *
                            0f32.max(intersection_y2 - intersection_y1);

    let box_a_area = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1);
    let box_b_area = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1);

    let union_area = box_a_area + box_b_area - intersection_area;

    if union_area == 0.0 {
        return 0.0;
    }

    intersection_area / union_area
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



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_overlapping_boxes() {
        let a = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
            class_id: 0,
            prob: 0.9,
        };

        let b = BoundingBox {
            x1: 2.0,
            y1: 2.0,
            x2: 3.0,
            y2: 3.0,
            class_id: 0,
            prob: 0.8,
        };

        let filtered_boxes = nms(&[a.clone(), b.clone()], 0.5);
        assert_eq!(filtered_boxes.len(), 2, "both boxes should be retained as they do not overlap.");

        assert_eq!(iou(&a, &b), 0.0, "the boxes do not overlap, so the IoU should be 0.");
    }

    #[test]
    fn test_overlapping_boxes_same_class() {
        let a = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 2.0,
            y2: 2.0,
            class_id: 0,
            prob: 0.9,
        };

        let b = BoundingBox {
            x1: 1.0,
            y1: 1.0,
            x2: 3.0,
            y2: 3.0,
            class_id: 0,
            prob: 0.8,
        };

        let expected_iou = 1.0 / 7.0;

        let filtered_boxes = nms(&[a.clone(), b.clone()], expected_iou - 0.1);
        assert_eq!(filtered_boxes.len(), 1, "only one box should be retained due to overlap.");
        assert_eq!(filtered_boxes[0].prob, 0.9, "the box with the higher probability should be retained.");

        assert_eq!((iou(&a, &b) - expected_iou).abs() < 1e-6, true, "the IoU should be approximately 1/7.");
    }

    #[test]
    fn test_overlapping_boxes_different_classes() {
        let a = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 2.0,
            y2: 2.0,
            class_id: 0,
            prob: 0.9,
        };

        let b = BoundingBox {
            x1: 1.0,
            y1: 1.0,
            x2: 3.0,
            y2: 3.0,
            class_id: 1,
            prob: 0.8,
        };

        let filtered_boxes = nms(&[a, b], 0.5);
        assert_eq!(filtered_boxes.len(), 2, "both boxes should be retained as they belong to different classes.");
    }

    #[test]
    fn test_iou_complete_overlap() {
        let a = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 2.0,
            y2: 2.0,
            class_id: 0,
            prob: 0.9,
        };

        let b = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 2.0,
            y2: 2.0,
            class_id: 0,
            prob: 0.8,
        };

        let expected_iou = 1.0;
        assert_eq!((iou(&a, &b) - expected_iou).abs() < 1e-6, true, "the IoU should be 1.0.");
    }

    #[test]
    fn test_iou_overlap_edge_case() {
        let a = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 2.0,
            y2: 2.0,
            class_id: 0,
            prob: 0.9,
        };

        let b = BoundingBox {
            x1: 2.0,
            y1: 2.0,
            x2: 4.0,
            y2: 4.0,
            class_id: 0,
            prob: 0.8,
        };

        let expected_iou = 0.0;
        assert_eq!((iou(&a, &b) - expected_iou).abs() < 1e-6, true, "the IoU should be 0.0.");
    }
}
