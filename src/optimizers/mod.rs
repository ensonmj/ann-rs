mod sgd;

pub use sgd::SGD;

pub trait Optimizer {
    fn optimize(
        &mut self,
        weights: &mut [Vec<f64>],
        bias: &mut [f64],
        gradients: &[Vec<f64>],
        bias_gradients: &[f64],
    );
}
