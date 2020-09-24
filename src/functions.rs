use rand::{thread_rng, Rng};
use std::cmp::Ordering;

pub fn into_onehot(idx: usize, classes: usize) -> Vec<f64> {
    debug_assert!(idx < classes, "onehot idx must less than classes");
    let mut vec = Vec::new();
    vec.resize(classes, f64::default());
    vec[idx] = 1.;
    vec
}

pub fn argmax(arr: &[f64]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap()
}

pub fn softmax(arr: &[f64]) -> Vec<f64> {
    let max = arr
        .iter()
        .max_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
        .unwrap();

    let exps: Vec<f64> = arr.iter().map(|x| (x - max).exp()).collect();
    let sum_exp: f64 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum_exp).collect()
}

pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

// matrix[expected][predicted]
// precision : p = tp / (tp + fp)
// recall    : r = tp / (tp + fn)
// accuracy  : acc = (tp + tn) / (tp + tn + fp + fn)
pub fn print_confusion_matrix(matrix: &Vec<Vec<u32>>) {
    let rows = matrix.len();
    let cols = matrix[0].len();
    assert_eq!(rows, cols);

    println!("confusion matrix: (p for precision)");
    // header
    println!(
        ">|{}",
        (0..cols).map(|i| format!("\t{}", i)).collect::<String>()
    );
    println!(
        "-|{}",
        (0..cols).map(|_| format!("\t-")).collect::<String>()
    );

    // precision
    println!(
        "p|{}",
        (0..cols)
            .map(|i| format!("\t{:.3}", {
                let tp = matrix[i][i] as f64;
                let all = matrix.iter().map(|vec| vec[i]).sum::<u32>() as f64;
                tp / all
            }))
            .collect::<String>()
    );
    println!(
        "-|{}",
        (0..cols).map(|_| format!("\t-")).collect::<String>()
    );

    // matrix
    let mut total_correct = 0;
    let mut total_ins = 0;
    for (predicted, row) in matrix.iter().enumerate() {
        print!("{}|", predicted);
        for (expected, num) in row.iter().enumerate() {
            if expected == predicted {
                print!("\t\x1B[32m{}\x1B[0m", num);
                total_correct += *num;
            } else {
                print!("\t{}", num);
            }
            total_ins += *num;
        }
        println!("");
    }

    // acc
    println!("Total instances: {}", total_ins);
    println!("Accuracy: {}", total_correct as f64 / total_ins as f64);
}

// https://zhuanlan.zhihu.com/p/25110150
// https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
// for f(x) = x or f(x) = tanh(x) (tanh(x) ~ x when x close to 0)
//
// create random weight matrix: vec![vec![float; in_dim]; out_dim]
pub fn xavier_init(in_dim: usize, out_dim: usize) -> Vec<Vec<f64>> {
    let variance = (6. / ((in_dim + out_dim) as f64)).sqrt();
    (0..out_dim)
        .map(|_| {
            (0..in_dim)
                .map(|_| thread_rng().gen_range(-variance, variance))
                .collect()
        })
        .collect()
}

// https://www.cnblogs.com/shine-lee/p/11908610.html
//
// for f(x) = ReLU(x)
pub fn he_init(in_dim: usize, out_dim: usize) -> Vec<Vec<f64>> {
    let variance = (2. / in_dim as f64).sqrt();
    // from caffe: use avg of in_dim + out_dim
    // let variance = (4. / (in_dim + out_dim) as f64).sqrt();
    (0..out_dim)
        .map(|_| {
            (0..in_dim)
                .map(|_| thread_rng().gen_range(-variance, variance))
                .collect()
        })
        .collect()
}
