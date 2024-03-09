use std::cmp::{
    max,
    min,
};

use bevy::{prelude::*, render::render_asset::RenderAssetUsages};
use image::{DynamicImage, GenericImageView, imageops::FilterType, ImageBuffer, Luma, RgbImage};
use ndarray::{Array, Array4, ArrayView4, Axis};


pub fn modnet_output_to_luma_image(
    output_value: &ort::Value,
) -> Image {
    let tensor: ort::Tensor<f32> = output_value.extract_tensor::<f32>().unwrap();

    let data = tensor.view();

    let shape = data.shape();
    let width = shape[3];
    let height = shape[2];

    let tensor_data = ArrayView4::from_shape((1, 1, height, width), data.as_slice().unwrap())
        .expect("Failed to create ArrayView4 from shape and data");

    let mut imgbuf = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let pixel_value = tensor_data[(0, 0, y as usize, x as usize)];
            let pixel_value = (pixel_value.clamp(0.0, 1.0) * 255.0) as u8;
            imgbuf.put_pixel(x as u32, y as u32, Luma([pixel_value]));
        }
    }

    let dyn_img = DynamicImage::ImageLuma8(imgbuf);

    Image::from_dynamic(dyn_img, false, RenderAssetUsages::all())
}


pub fn image_to_modnet_input(
    image: &Image,
) -> Array4<f32> {
    assert_eq!(image.texture_descriptor.format, bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb);

    let ref_size = 512;
    let (
        x_scale,
        y_scale,
    ) = get_scale_factor(
        image.height(),
        image.width(),
        ref_size,
    );

    let resized_image = resize_image(
        &image.clone().try_into_dynamic().unwrap(),
        x_scale,
        y_scale,
    );

    image_to_ndarray(&resized_image)
}


fn get_scale_factor(im_h: u32, im_w: u32, ref_size: u32) -> (f32, f32) {
    let mut im_rh;
    let mut im_rw;

    if max(im_h, im_w) < ref_size || min(im_h, im_w) > ref_size {
        if im_w >= im_h {
            im_rh = ref_size;
            im_rw = (im_w as f32 / im_h as f32 * ref_size as f32) as u32;
        } else {
            im_rw = ref_size;
            im_rh = (im_h as f32 / im_w as f32 * ref_size as f32) as u32;
        }
    } else {
        im_rh = im_h;
        im_rw = im_w;
    }

    im_rw = im_rw - im_rw % 32;
    im_rh = im_rh - im_rh % 32;

    (im_rw as f32 / im_w as f32, im_rh as f32 / im_h as f32)
}


fn image_to_ndarray(img: &RgbImage) -> Array4<f32> {
    let (width, height) = img.dimensions();

    // convert RgbImage to a Vec of f32 values normalized to [-1, 1]
    let raw: Vec<f32> = img.pixels()
        .flat_map(|p| {
            p.0.iter().map(|&e| {
                (e as f32 - 127.5) / 127.5
            })
        })
        .collect();

    // create a 3D array from the raw pixel data
    let arr = Array::from_shape_vec((height as usize, width as usize, 3), raw)
        .expect("failed to create ndarray from image raw data");

    // rearrange the dimensions from [height, width, channels] to [1, channels, height, width]
    let arr = arr.permuted_axes([2, 0, 1]).insert_axis(Axis(0));

    arr
}

fn resize_image(image: &DynamicImage, x_scale: f32, y_scale: f32) -> RgbImage {
    let (width, height) = image.dimensions();
    let new_width = (width as f32 * x_scale) as u32;
    let new_height = (height as f32 * y_scale) as u32;

    image.resize_exact(new_width, new_height, FilterType::Triangle).to_rgb8()
}
