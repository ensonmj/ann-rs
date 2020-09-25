use super::Optimizer;
use crate::functions::{for_each, transform};

pub struct Adam {
    pub learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    count: u64,
    means: Vec<Vec<Vec<f64>>>,
    bias_means: Vec<Vec<f64>>,
    virances: Vec<Vec<Vec<f64>>>,
    bias_virances: Vec<Vec<f64>>,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Adam {
        Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 0.00000001,
            count: 0,
            means: vec![],
            bias_means: vec![],
            virances: vec![],
            bias_virances: vec![],
        }
    }

    // init mean and virance default 0
    fn init_layer_mean_and_virance(&mut self, gradients: &[Vec<f64>], bias_gradients: &[f64]) {
        let mean: Vec<Vec<f64>> = gradients
            .iter()
            .map(|row| row.iter().map(|_| 0.).collect())
            .collect();
        let bias_mean: Vec<f64> = bias_gradients.iter().map(|_| 0.).collect();

        let virance = mean.clone();
        let bias_virance = bias_mean.clone();

        self.means.push(mean);
        self.bias_means.push(bias_mean);
        self.virances.push(virance);
        self.bias_virances.push(bias_virance);
    }
}

// https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
// https://blog.csdn.net/yzy_1996/article/details/84618536
// https://zh.d2l.ai/chapter_optimization/adam.html
impl Optimizer for Adam {
    fn optimize(
        &mut self,
        idx: usize,
        weights: &mut [Vec<f64>],
        bias: &mut [f64],
        gradients: &mut [Vec<f64>],
        bias_gradients: &mut [f64],
    ) {
        // increased after every all layers updated
        if idx == 0 {
            self.count += 1;
        }

        if self.means.len() <= idx {
            self.init_layer_mean_and_virance(gradients, bias_gradients);
        }

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let param = self.count as f64;

        // step1. mean(t) = beta1 * mean(t-1) + (1 - beta1) * gradient(t)
        transform(
            &mut self.means[idx],
            gradients,
            &mut self.bias_means[idx],
            bias_gradients,
            |mean, gradient| *mean = beta1 * *mean + (1. - beta1) * gradient,
        );

        // step2. viranece(t) = beta2 * virance(t-1) + (1 - beta2) * gradient(t)^2
        transform(
            &mut self.virances[idx],
            gradients,
            &mut self.bias_virances[idx],
            bias_gradients,
            |mean, gradient| *mean = beta2 * *mean + (1. - beta2) * gradient.powf(2.),
        );

        // step3. mean_bias_corr(t) = mean(t) / (1 - beta1^param(t))
        let (mut corr_mean, mut bias_corr_mean) =
            for_each(&self.means[idx], &self.bias_means[idx], |mean| {
                mean / (1. - beta1.powf(param))
            });

        // step4. virance_bias_corr(t) = virance(t) / (1 - beta2^param(t))
        let (corr_virance, bias_corr_virance) =
            for_each(&self.virances[idx], &self.bias_virances[idx], |virance| {
                virance / (1. - beta2.powf(param))
            });

        // step5. gradient(t) = mean_bias_corr(t) / (virance_bias_corr(t).sqrt() + eps)
        transform(
            &mut corr_mean,
            &corr_virance,
            &mut bias_corr_mean,
            &bias_corr_virance,
            |mean, virance| *mean = *mean / (virance.sqrt() + self.eps),
        );

        // step6. update weights and bias
        // use corr_mean as gradient, bias_corr_mean as bias_gradient
        transform(
            weights,
            &corr_mean, // &gradients,
            bias,
            &bias_corr_mean, // &bias_gradients,
            |mut_weight_or_bias, gradient| *mut_weight_or_bias -= self.learning_rate * gradient,
        );
    }
}
