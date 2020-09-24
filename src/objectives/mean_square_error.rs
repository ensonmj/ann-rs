use super::Objective;
use crate::activators::{Activator, Sigmoid};

pub struct MeanSquareError;

impl MeanSquareError {
    pub fn new() -> MeanSquareError {
        MeanSquareError {}
    }
}

impl Objective<Sigmoid> for MeanSquareError {
    fn loss(&self, predict: &[Vec<f64>], expected: &[Vec<f64>]) -> Vec<f64> {
        Sigmoid
            .activate(predict)
            .iter()
            .zip(expected.iter())
            .map(|(predict, expected)| {
                predict
                    .iter()
                    .zip(expected.iter())
                    .map(|(predict, expected)| 0.5 * (expected - predict).powf(2.))
                    .sum()
            })
            .collect()
    }

    fn delta_without_deriv(&self, predict: &[Vec<f64>], expected: &[Vec<f64>]) -> Vec<Vec<f64>> {
        Sigmoid
            .activate(predict)
            .iter()
            .zip(expected.iter())
            .map(|(predict, expected)| {
                predict
                    .iter()
                    .zip(expected.iter())
                    .map(|(predict, expected)| (predict - expected) * predict * (1. - predict))
                    .collect()
            })
            .collect()
    }

    fn predict_from_logits(&self, logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
        Sigmoid.activate(logits)
    }
}
