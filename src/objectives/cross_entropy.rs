use super::Objective;
use crate::activators::{Activator, Softmax};
use crate::functions::{argmax, into_onehot};

pub struct CrossEntropy;

impl CrossEntropy {
    pub fn new() -> CrossEntropy {
        CrossEntropy {}
    }
}

impl Objective<Softmax> for CrossEntropy {
    // loss = -SUM(expected(i) * ln(Softmax(predict(i))))
    fn loss(&self, predict: &[f64], expected: &[f64]) -> f64 {
        Softmax
            .activate(predict)
            .iter()
            .zip(expected.iter())
            .map(|(predict, expected)| -(expected * predict.ln()))
            .sum()
    }

    // https://zhuanlan.zhihu.com/p/25723112
    // http://blog.prince2015.club/2020/03/27/softmax/
    fn delta_without_deriv(&self, predict: &[f64], expected: &[f64]) -> Vec<f64> {
        // if expected[j] == 1
        // for i: 0-n
        //     if i == j, expected=1: delta = (predict-1) <- (predict-expected)
        //     if i != j, expected=0: delta = predict     <- (predict-expected)
        Softmax
            .activate(predict)
            .iter()
            .zip(expected.iter())
            .map(|(predict, expected)| predict - expected)
            .collect()
    }

    fn predict_from_logits(&self, logits: &[f64]) -> Vec<f64> {
        into_onehot(argmax(&logits), logits.len())
    }
}
