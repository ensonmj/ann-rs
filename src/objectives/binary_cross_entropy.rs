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
        assert_eq!(
            expected.len(),
            1,
            "binary cross entropy should have only one dimension"
        );
        assert_eq!(
            predict.len(),
            1,
            "binary cross entropy result should have only one dimension"
        );
        let s = Sigmoid.activate(predict);
        expected
            .iter()
            .zip(s.iter())
            .map(|(&expected, predict)| {
                -(if expected < 1e-6 {
                    (1.0 - predict).ln()
                } else {
                    predict.ln()
                })
            })
            .sum()
    }

    fn delta_without_deriv(&self, predict: &[f64], expected: &[f64]) -> Vec<f64> {
        let s = Sigmoid.activate(predict);
        expected
            .iter()
            .zip(s.iter())
            .map(|(expected, predict)| expected * (predict - 1.))
            .collect()
    }

    fn predict_from_probs(&self, probs: &[f64]) -> f64 {
        if probs[0] >= 0.5 {
            1.
        } else {
            0.
        }
    }
}
