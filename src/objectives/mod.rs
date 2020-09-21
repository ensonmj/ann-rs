use crate::activators::Activator;

mod binary_cross_entropy;
mod cross_entropy;
mod mean_square_error;

pub use binary_cross_entropy::BinaryCrossEntropy;
pub use cross_entropy::CrossEntropy;
pub use mean_square_error::MeanSquareError;

pub trait Objective<A: Activator> {
    fn loss(&self, predict: &[f64], expected: &[f64]) -> f64;
    fn delta_without_deriv(&self, predict: &[f64], expected: &[f64]) -> Vec<f64>;
    fn predict_from_probs(&self, probs: &[f64]) -> f64;
}
