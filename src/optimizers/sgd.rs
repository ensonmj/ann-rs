use super::Optimizer;
use crate::functions::transform;

pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> SGD {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    fn optimize(
        &mut self,
        _idx: usize,
        weights: &mut [Vec<f64>],
        bias: &mut [f64],
        gradients: &mut [Vec<f64>],
        bias_gradients: &mut [f64],
    ) {
        transform(
            weights,
            gradients,
            bias,
            bias_gradients,
            |weight_or_bias, gradient| *weight_or_bias -= self.learning_rate * gradient,
        );
    }
}
