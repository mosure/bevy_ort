use bevy::{prelude::*, render::render_asset::RenderAssetUsages};
use image::{DynamicImage, GenericImageView, imageops::FilterType, ImageBuffer, Luma, RgbImage};
use ndarray::{Array, Array4, ArrayView4, Axis, s};


pub fn modnet_output_to_luma_images(
    output_value: &ort::Value,
) -> Vec<Image> {
    let data = output_value.extract_tensor::<f32>().unwrap();

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
    max_size: Option<(u32, u32)>,
) -> Array4<f32> {
    if images.is_empty() {
        panic!("no images provided");
    }

    let ref_size = 512;
    let &first_image = images.first().unwrap();

    let image = first_image.to_owned();

    let (x_scale, y_scale) = get_scale_factor(image.height(), image.width(), ref_size, max_size);
    let resized_image = resize_image(&image.try_into_dynamic().unwrap(), x_scale, y_scale);
    let first_image_ndarray = image_to_ndarray(&resized_image);

    let single_image_shape = first_image_ndarray.dim();
    let n_images = images.len();
    let batch_shape = (n_images, single_image_shape.1, single_image_shape.2, single_image_shape.3);

    let mut aggregate = Array4::<f32>::zeros(batch_shape);

    for (i, &image) in images.iter().enumerate() {
        let image = image.to_owned();
        let (x_scale, y_scale) = get_scale_factor(image.height(), image.width(), ref_size, max_size);
        let resized_image = resize_image(&image.try_into_dynamic().unwrap(), x_scale, y_scale);
        let image_ndarray = image_to_ndarray(&resized_image);

        let slice = s![i, .., .., ..];
        aggregate.slice_mut(slice).assign(&image_ndarray.index_axis_move(Axis(0), 0));
    }

    aggregate
}


fn get_scale_factor(im_h: u32, im_w: u32, ref_size: u32, max_size: Option<(u32, u32)>) -> (f32, f32) {
    // Calculate the scale factor based on the maximum size constraints
    let scale_factor_max = max_size.map_or(1.0, |(max_w, max_h)| {
        f32::min(max_w as f32 / im_w as f32, max_h as f32 / im_h as f32)
    });

    // Calculate the target dimensions after applying the max scale factor (clipping to max_size)
    let (target_h, target_w) = ((im_h as f32 * scale_factor_max).round() as u32, (im_w as f32 * scale_factor_max).round() as u32);

    // Calculate the scale factor to fit within the reference size, considering the target dimensions
    let (scale_factor_ref_w, scale_factor_ref_h) = if std::cmp::max(target_h, target_w) < ref_size {
        let scale_factor = ref_size as f32 / std::cmp::max(target_h, target_w) as f32;
        (scale_factor, scale_factor)
    } else {
        (1.0, 1.0) // Do not upscale if target dimensions are within reference size
    };

    // Calculate the final scale factor as the minimum of the max scale factor and the reference scale factor
    let final_scale_w = f32::min(scale_factor_max, scale_factor_ref_w);
    let final_scale_h = f32::min(scale_factor_max, scale_factor_ref_h);

    // Adjust dimensions to ensure they are multiples of 32
    let final_w = ((im_w as f32 * final_scale_w).round() as u32) - ((im_w as f32 * final_scale_w).round() as u32) % 32;
    let final_h = ((im_h as f32 * final_scale_h).round() as u32) - ((im_h as f32 * final_scale_h).round() as u32) % 32;

    // Return the scale factors based on the original image dimensions
    (final_w as f32 / im_w as f32, final_h as f32 / im_h as f32)
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
