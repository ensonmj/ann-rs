use byteorder::{BigEndian, ReadBytesExt};
use log;
use pretty_env_logger;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::fs::File;
use std::io::Cursor;
use std::io::{BufReader, Error, Read};

use ann_rs::activators::Relu;
use ann_rs::functions::*;
use ann_rs::objectives::CrossEntropy;
use ann_rs::optimizers::SGD;
use ann_rs::NetworkBuilder;

fn print_sample_image(image: &[u8], rows: usize, cols: usize, label: u8) {
    // Check that the image isn't empty and has a valid number of rows.
    assert_ne!(image.len(), 0, "There are no pixels in this image.");
    assert_eq!(
        image.len() % rows,
        0,
        "Number of pixels does not evenly divide into number of rows."
    );

    println!("Sample image:");
    // Print each row.
    for row in 0..rows {
        for col in 0..cols {
            if image[row * cols + col] == 0 {
                print!("__");
            } else {
                print!("##");
            }
        }
        print!("\n");
    }
    println!("Sample image label: {}", label);
}

fn read_image_and_labels(
    image_file: &str,
    label_file: &str,
) -> Result<(Vec<f64>, Vec<u8>, usize, usize, usize), Error> {
    let mut reader = BufReader::new(File::open(image_file)?);
    let mut header = [0u8; 4 * 4]; // four u32
    reader.read_exact(&mut header)?;
    let mut rdr = Cursor::new(header);
    let magic_nr = rdr.read_u32::<BigEndian>()?;
    let size = rdr.read_u32::<BigEndian>()?;
    let rows = rdr.read_u32::<BigEndian>()?;
    let cols = rdr.read_u32::<BigEndian>()?;
    let mut image_data = Vec::<u8>::with_capacity(size as usize);
    reader.read_to_end(&mut image_data)?;
    log::debug!(
        "train data: {}-{}-{}-{}-{}",
        magic_nr,
        size,
        rows,
        cols,
        image_data.len(),
    );

    let mut reader = BufReader::new(File::open(label_file)?);
    let mut header = [0u8; 4 * 2]; // two u32
    reader.read_exact(&mut header)?;
    let mut rdr = Cursor::new(header);
    let magic_nr = rdr.read_u32::<BigEndian>()?;
    let size = rdr.read_u32::<BigEndian>()?;
    let mut label_data = Vec::<u8>::with_capacity(size as usize);
    reader.read_to_end(&mut label_data)?;
    log::debug!("label data: {}-{}-{}", magic_nr, size, label_data.len(),);

    let sample = thread_rng().gen_range(0, size);
    let (start, end): (usize, usize) = (
        (sample * rows * cols) as usize,
        ((sample + 1) * rows * cols) as usize,
    );
    print_sample_image(
        &image_data[start..end],
        rows as usize,
        cols as usize,
        label_data[sample as usize],
    );

    // Normalize the image.
    let image_data: Vec<f64> = image_data
        .into_iter()
        .map(|pixel| 2.0 * f64::from(pixel) / 255.0 - 1.0)
        .collect();

    Ok((
        image_data,
        label_data,
        size as usize,
        rows as usize,
        cols as usize,
    ))
}

fn main() -> Result<(), Error> {
    pretty_env_logger::init();

    let train_image_file = "./data/train-images-idx3-ubyte";
    let train_label_file = "./data/train-labels-idx1-ubyte";
    let (image_data, label_data, size, rows, cols) =
        read_image_and_labels(train_image_file, train_label_file)?;

    // create a network with 3 layers:
    let mut nn = NetworkBuilder::new()
        .input(rows * cols)
        .add_layer(300, Box::new(Relu))
        .add_layer(300, Box::new(Relu))
        .output(10)
        .minimize_to(CrossEntropy::new())
        .optimize_with(SGD::new(0.0001))
        .build();

    // for train data and labels shuffle
    let mut idxs: Vec<usize> = (0..size).collect();
    for i in 0..5 {
        let mut errors = vec![];
        idxs.shuffle(&mut thread_rng());
        for (j, idx) in idxs.iter().enumerate() {
            let input = &image_data[idx * rows * cols..(idx + 1) * rows * cols];
            let label = into_onehot(label_data[*idx]);
            errors.push(nn.fit(&input, &label));
            if j % 1000 == 0 {
                let mean_errors = errors.iter().sum::<f64>() / (errors.len() as f64);
                log::info!("{}-{}: {}", i, j, mean_errors);
            }
        }
    }

    // test
    let test_image_file = "./data/t10k-images-idx3-ubyte";
    let test_label_file = "./data/t10k-labels-idx1-ubyte";
    let (test_image_data, test_label_data, _, _, _) =
        read_image_and_labels(test_image_file, test_label_file)?;

    let mut confusion_matrix: Vec<Vec<u32>> = vec![vec![0u32; 10]; 10];
    for (i, input) in test_image_data.chunks(rows * cols).enumerate() {
        let infer = nn.infer(input) as usize;
        let label = test_label_data[i] as usize;
        confusion_matrix[infer][label] += 1;
    }
    print_confusion_matrix(&confusion_matrix);

    Ok(())
}
