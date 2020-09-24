use super::Optimizer;

pub struct SGD {
    pub learning_rate: f64,
    pub decay_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64, decay_rate: f64) -> SGD {
        SGD {
            learning_rate,
            decay_rate,
        }
    }
}

impl Optimizer for SGD {
    fn optimize(
        &mut self,
        weights: &mut [Vec<f64>],
        bias: &mut [f64],
        gradients: &[Vec<f64>],
        bias_gradients: &[f64],
    ) {
        log::debug!("learning rate: {}", &self.learning_rate);
        log::debug!("before weights: {:?}, gradient: {:?}", &weights, &gradients);
        weights
            .iter_mut()
            .zip(gradients.iter())
            .for_each(|(ws, gs)| {
                ws.iter_mut()
                    .zip(gs.iter())
                    .for_each(|(w, g)| *w -= self.learning_rate * g)
            });
        log::debug!("after weights: {:?}", &weights);
        log::debug!("before bias: {:?}, gradient: {:?}", &bias, &bias_gradients);
        bias.iter_mut()
            .zip(bias_gradients.iter())
            .for_each(|(bias, gradient)| *bias -= self.learning_rate * gradient);
        log::debug!("after bias: {:?}", &bias);

        // self.learning_rate *= self.decay_rate;
    }
}
