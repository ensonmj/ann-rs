use super::Objective;
use crate::activators::{Activator, Sigmoid};

pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> BinaryCrossEntropy {
        BinaryCrossEntropy {}
    }
}

impl Objective<Sigmoid> for BinaryCrossEntropy {
    fn loss(&self, predict: &[f64], expected: &[f64]) -> f64 {
        debug_assert_eq!(
            expected.len(),
            1,
            "binary cross entropy should have only one dimension"
        );
        debug_assert_eq!(
            predict.len(),
            1,
            "binary cross entropy result should have only one dimension"
        );
        Sigmoid
            .activate(predict)
            .iter()
            .zip(expected.iter())
            .map(|(predict, &expected)| {
                -(if expected < 1e-6 {
                    (1.0 - predict).ln()
                } else {
                    predict.ln()
                })
            })
            .sum()
    }

    // https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
    fn delta_without_deriv(&self, predict: &[f64], expected: &[f64]) -> Vec<f64> {
        Sigmoid
            .activate(predict)
            .iter()
            .zip(expected.iter())
            .map(|(predict, expected)| predict - expected)
            .collect()
    }

    fn predict_from_logits(&self, logits: &[f64]) -> Vec<f64> {
        Sigmoid
            .activate(logits)
            .iter()
            .map(|&v| if v > 0.5 { 1. } else { 0. })
            .collect()
    }
}
