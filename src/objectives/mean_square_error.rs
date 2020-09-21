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
        let s = Sigmoid.activate(predict);
        expected
            .iter()
            .zip(s.iter())
            .map(|(expected, predict)| 0.5 * (expected - predict).powf(2.))
            .sum()
    }

    fn delta_without_deriv(&self, predict: &[f64], expected: &[f64]) -> Vec<f64> {
        let s = Sigmoid.activate(predict);
        expected
            .iter()
            .zip(s.iter())
            .map(|(expected, predict)| (predict - expected) * predict * (1. - predict))
            .collect()
    }

    fn predict_from_probs(&self, probs: &[f64]) -> f64 {
        let s = Sigmoid.activate(probs);
        s[0]
    }
}
