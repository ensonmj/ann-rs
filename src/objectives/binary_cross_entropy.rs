use super::Objective;
use crate::activators::{Activator, Sigmoid};

pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> BinaryCrossEntropy {
        BinaryCrossEntropy {}
    }
}

impl Objective<Sigmoid> for BinaryCrossEntropy {
    fn loss(&self, predict: &[Vec<f64>], expected: &[Vec<f64>]) -> Vec<f64> {
        Sigmoid
            .activate(predict)
            .iter()
            .zip(expected.iter())
            .map(|(predict, expected)| {
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
                predict
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
            })
            .collect()
    }

    // https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
    fn delta_without_deriv(&self, predict: &[Vec<f64>], expected: &[Vec<f64>]) -> Vec<Vec<f64>> {
        Sigmoid
            .activate(predict)
            .iter()
            .zip(expected.iter())
            .map(|(predict, expected)| {
                predict
                    .iter()
                    .zip(expected.iter())
                    .map(|(predict, &expected)| predict - expected)
                    .collect()
            })
            .collect()
    }

    fn predict_from_logits(&self, logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
        Sigmoid
            .activate(logits)
            .iter()
            .map(|v| v.iter().map(|&v| if v > 0.5 { 1. } else { 0. }).collect())
            .collect()
    }
}
