use std::cmp::{
    max,
    min,
};

use bevy::{prelude::*, render::render_asset::RenderAssetUsages};
use image::{DynamicImage, GenericImageView, imageops::FilterType, ImageBuffer, Luma, RgbImage};
use ndarray::{Array, Array4, ArrayView4, Axis, s};


pub fn modnet_output_to_luma_images(
    output_value: &ort::Value,
) -> Vec<Image> {
    let tensor: ort::Tensor<f32> = output_value.extract_tensor::<f32>().unwrap();

    let data = tensor.view();

    let shape = data.shape();
    let batch_size = shape[0];
    let width = shape[3];
    let height = shape[2];

    let tensor_data = ArrayView4::from_shape((batch_size, 1, height, width), data.as_slice().unwrap())
        .expect("failed to create ArrayView4 from shape and data");

    let mut images = Vec::new();

    for i in 0..batch_size {
        let mut imgbuf = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let pixel_value = tensor_data[(i, 0, y, x)];
                let pixel_value = (pixel_value.clamp(0.0, 1.0) * 255.0) as u8;
                imgbuf.put_pixel(x as u32, y as u32, Luma([pixel_value]));
            }
        }

        let dyn_img = DynamicImage::ImageLuma8(imgbuf);

        images.push(Image::from_dynamic(dyn_img, false, RenderAssetUsages::all()));
    }

    images
}

pub fn images_to_modnet_input(
    images: Vec<&Image>,
) -> Array4<f32> {
    // TODO: better error handling
    if images.is_empty() {
        panic!("no images provided");
    }

    let ref_size = 512;

    let &first_image = images.first().unwrap();
    assert_eq!(first_image.texture_descriptor.format, bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb);

    let dynamic_image = first_image.clone().try_into_dynamic().unwrap();
    let (x_scale, y_scale) = get_scale_factor(dynamic_image.height(), dynamic_image.width(), ref_size);
    let resized_image = resize_image(&dynamic_image, x_scale, y_scale);
    let first_image_ndarray = image_to_ndarray(&resized_image);
    let single_image_shape = first_image_ndarray.dim();
    let n_images = images.len();
    let batch_shape = (n_images, single_image_shape.1, single_image_shape.2, single_image_shape.3);

    let mut aggregate = Array4::<f32>::zeros(batch_shape);

    for (i, &image) in images.iter().enumerate() {
        let dynamic_image = image.clone().try_into_dynamic().unwrap();
        let (x_scale, y_scale) = get_scale_factor(dynamic_image.height(), dynamic_image.width(), ref_size);
        let resized_image = resize_image(&dynamic_image, x_scale, y_scale);
        let image_ndarray = image_to_ndarray(&resized_image);

        let slice = s![i, .., .., ..];
        aggregate.slice_mut(slice).assign(&image_ndarray.index_axis_move(Axis(0), 0));
    }

    aggregate
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
    arr.permuted_axes([2, 0, 1]).insert_axis(Axis(0))
}

fn resize_image(image: &DynamicImage, x_scale: f32, y_scale: f32) -> RgbImage {
    let (width, height) = image.dimensions();
    let new_width = (width as f32 * x_scale) as u32;
    let new_height = (height as f32 * y_scale) as u32;

    image.resize_exact(new_width, new_height, FilterType::Triangle).to_rgb8()
}
