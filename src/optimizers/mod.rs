mod sgd;

pub use sgd::SGD;

pub trait Optimizer {
    // weights: one layer's weights
    // bias: one layer's bias
    // gradients: one layer's gradients,
    // bias_gradients: one layer's bias_gradients,
    fn optimize(
        &mut self,
        weights: &mut [Vec<f64>],
        bias: &mut [f64],
        gradients: &[Vec<f64>],
        bias_gradients: &[f64],
    );
}
