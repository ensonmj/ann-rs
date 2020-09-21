use super::Optimizer;

pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> SGD {
        SGD {
            learning_rate: learning_rate,
        }
    }
}

impl Optimizer for SGD {
    fn optimize(
        &self,
        weights: &mut [Vec<f64>],
        bias: &mut [f64],
        gradients: &[Vec<f64>],
        bias_gradients: &[f64],
    ) {
        weights
            .iter_mut()
            .zip(gradients.iter())
            .for_each(|(ws, gs)| {
                ws.iter_mut()
                    .zip(gs.iter())
                    .for_each(|(w, g)| *w -= self.learning_rate * g)
            });
        bias.iter_mut()
            .zip(bias_gradients.iter())
            .for_each(|(bias, gradient)| *bias -= self.learning_rate * gradient);
    }
}