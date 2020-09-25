mod adam;
mod sgd;

pub use adam::Adam;
pub use sgd::SGD;

pub trait Optimizer {
    // idx: layer index
    // weights: one layer's weights
    // bias: one layer's bias
    // gradients: one layer's gradients,
    // bias_gradients: one layer's bias_gradients,
    fn optimize(
        &mut self,
        idx: usize,
        weights: &mut [Vec<f64>],
        bias: &mut [f64],
        gradients: &mut [Vec<f64>],
        bias_gradients: &mut [f64],
    );
}
