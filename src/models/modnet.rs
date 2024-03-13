use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{
            Extent3d,
            TextureDimension,
            TextureFormat,
        },
    },
};
use image::{DynamicImage, GenericImageView, imageops::FilterType, ImageBuffer, Luma, RgbImage};
use ndarray::{Array, Array4, ArrayView4};
use rayon::prelude::*;


pub fn modnet_output_to_luma_images(
    output_value: &ort::Value,
) -> Vec<Image> {
    let tensor = output_value.extract_tensor::<f32>().unwrap();
    let data = tensor.view();

    let shape = data.shape();
    let batch_size = shape[0];
    let width = shape[3];
    let height = shape[2];

    let tensor_data = ArrayView4::from_shape((batch_size, 1, height, width), data.as_slice().unwrap())
        .expect("failed to create ArrayView4 from shape and data");

    let images = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let mut imgbuf = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width as u32, height as u32);

            for y in 0..height {
                for x in 0..width {
                    let pixel_value = tensor_data[(i, 0, y, x)];
                    let pixel_value = (pixel_value.clamp(0.0, 1.0) * 255.0) as u8;
                    imgbuf.put_pixel(x as u32, y as u32, Luma([pixel_value]));
                }
            }

            let image = Image::new(
                Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                imgbuf.into_raw(),
                TextureFormat::R8Unorm,
                RenderAssetUsages::all(),
            );

            image
        })
        .collect::<Vec<_>>();

    images
}


pub fn images_to_modnet_input(
    images: &[&Image],
    max_size: Option<(u32, u32)>,
) -> Array4<f32> {
    if images.is_empty() {
        panic!("no images provided");
    }

    let ref_size = 512;
    let &first_image = images.first().unwrap();

    let (x_scale, y_scale) = get_scale_factor(first_image.height(), first_image.width(), ref_size, max_size);

    let processed_images: Vec<Array4<f32>> = images
        .par_iter()
        .map(|&image| {
            let resized_image = resize_image(&image.clone().try_into_dynamic().unwrap(), x_scale, y_scale);
            image_to_ndarray(&resized_image)
        })
        .collect();

    let aggregate = Array::from_shape_vec(
        (processed_images.len(), processed_images[0].shape()[1], processed_images[0].shape()[2], processed_images[0].shape()[3]),
        processed_images.iter().flat_map(|a| a.iter().cloned()).collect(),
    ).unwrap();

    aggregate
}


fn get_scale_factor(im_h: u32, im_w: u32, ref_size: u32, max_size: Option<(u32, u32)>) -> (f32, f32) {
    let scale_factor_max = max_size.map_or(1.0, |(max_w, max_h)| {
        f32::min(max_w as f32 / im_w as f32, max_h as f32 / im_h as f32)
    });

    let (target_h, target_w) = ((im_h as f32 * scale_factor_max).round() as u32, (im_w as f32 * scale_factor_max).round() as u32);

    let (scale_factor_ref_w, scale_factor_ref_h) = if std::cmp::max(target_h, target_w) < ref_size {
        let scale_factor = ref_size as f32 / std::cmp::max(target_h, target_w) as f32;
        (scale_factor, scale_factor)
    } else {
        (1.0, 1.0)
    };

    let final_scale_w = f32::min(scale_factor_max, scale_factor_ref_w);
    let final_scale_h = f32::min(scale_factor_max, scale_factor_ref_h);

    let final_w = ((im_w as f32 * final_scale_w).round() as u32) - ((im_w as f32 * final_scale_w).round() as u32) % 32;
    let final_h = ((im_h as f32 * final_scale_h).round() as u32) - ((im_h as f32 * final_scale_h).round() as u32) % 32;

    (final_w as f32 / im_w as f32, final_h as f32 / im_h as f32)
}


fn image_to_ndarray(img: &RgbImage) -> Array4<f32> {
    let (width, height) = img.dimensions();

    let arr = Array::from_shape_fn((1, 3, height as usize, width as usize), |(_, c, y, x)| {
        let pixel = img.get_pixel(x as u32, y as u32);
        let channel_value = match c {
            0 => pixel[0],
            1 => pixel[1],
            2 => pixel[2],
            _ => unreachable!(),
        };
        (channel_value as f32 - 127.5) / 127.5
    });

    arr
}

fn resize_image(image: &DynamicImage, x_scale: f32, y_scale: f32) -> RgbImage {
    let (width, height) = image.dimensions();
    let new_width = (width as f32 * x_scale) as u32;
    let new_height = (height as f32 * y_scale) as u32;

    image.resize_exact(new_width, new_height, FilterType::Triangle).to_rgb8()
}
