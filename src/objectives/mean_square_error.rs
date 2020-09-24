use super::Objective;
use crate::activators::{Activator, Sigmoid};

pub struct MeanSquareError;

impl MeanSquareError {
    pub fn new() -> MeanSquareError {
        MeanSquareError {}
    }
}

impl Objective<Sigmoid> for MeanSquareError {
    fn loss(&self, predict: &[f64], expected: &[f64]) -> f64 {
        Sigmoid
            .activate(predict)
            .iter()
            .zip(expected.iter())
            .map(|(predict, expected)| 0.5 * (expected - predict).powf(2.))
            .sum()
    }

    fn delta_without_deriv(&self, predict: &[f64], expected: &[f64]) -> Vec<f64> {
        Sigmoid
            .activate(predict)
            .iter()
            .zip(expected.iter())
            .map(|(predict, expected)| (predict - expected) * predict * (1. - predict))
            .collect()
    }

    fn predict_from_logits(&self, logits: &[f64]) -> Vec<f64> {
        Sigmoid.activate(logits)
    }
}
