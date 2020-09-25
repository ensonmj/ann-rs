use super::Optimizer;

pub struct Adam {
    pub learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    count: u64,
    gradient_mts: Vec<Vec<Vec<f64>>>,
    bias_gradient_mts: Vec<Vec<f64>>,
    gradient_vts: Vec<Vec<Vec<f64>>>,
    bias_gradient_vts: Vec<Vec<f64>>,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Adam {
        Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 0.00000001,
            count: 0,
            gradient_mts: vec![],
            bias_gradient_mts: vec![],
            gradient_vts: vec![],
            bias_gradient_vts: vec![],
        }
    }
}

impl Optimizer for Adam {
    fn optimize(
        &mut self,
        idx: usize,
        weights: &mut [Vec<f64>],
        bias: &mut [f64],
        gradients: &[Vec<f64>],
        bias_gradients: &[f64],
    ) {
        if idx == 0 {
            // increased after every all layers
            self.count += 1;
        }

        // https://zh.d2l.ai/chapter_optimization/adam.html
        // https://blog.csdn.net/yzy_1996/article/details/84618536
        if self.gradient_mts.len() <= idx {
            // create mt and vt cache
            let gradient_mt: Vec<Vec<f64>> = gradients
                .iter()
                .map(|row| row.iter().map(|_| 0.).collect())
                .collect();
            let bias_gradient_mt: Vec<f64> = bias_gradients.iter().map(|_| 0.).collect();

            let gradient_vt = gradient_mt.clone();
            let bias_gradient_vt = bias_gradient_mt.clone();

            self.gradient_mts.push(gradient_mt);
            self.bias_gradient_mts.push(bias_gradient_mt);
            self.gradient_vts.push(gradient_vt);
            self.bias_gradient_vts.push(bias_gradient_vt);
        }

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let count = self.count as f64;
        let gradient_mt = &mut self.gradient_mts[idx];
        let gradient_vt = &mut self.gradient_vts[idx];
        let bias_gradient_mt = &mut self.bias_gradient_mts[idx];
        let bias_gradient_vt = &mut self.bias_gradient_vts[idx];

        // v[:] = beta1 * v + (1 - beta1) * p.grad
        gradient_mt
            .iter_mut()
            .zip(gradients.iter())
            .for_each(|(mt_row, g_row)| {
                mt_row
                    .iter_mut()
                    .zip(g_row.iter())
                    .for_each(|(mt_col, g_col)| *mt_col = beta1 * *mt_col + (1. - beta1) * *g_col)
            });
        bias_gradient_mt
            .iter_mut()
            .zip(bias_gradients.iter())
            .for_each(|(mt_row, g_row)| *mt_row = beta1 * *mt_row + (1. - beta1) * *g_row);

        // s[:] = beta2 * s + (1 - beta2) * p.grad.square()
        gradient_vt
            .iter_mut()
            .zip(gradients.iter())
            .for_each(|(vt_row, g_row)| {
                vt_row
                    .iter_mut()
                    .zip(g_row.iter())
                    .for_each(|(vt_col, g_col)| {
                        *vt_col = beta2 * *vt_col + (1. - beta2) * g_col.powf(2.)
                    })
            });
        bias_gradient_vt
            .iter_mut()
            .zip(bias_gradients.iter())
            .for_each(|(vt_row, g_row)| *vt_row = beta2 * *vt_row + (1. - beta2) * g_row.powf(2.));

        // v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        let gradient_mtt: Vec<Vec<f64>> = gradient_mt
            .iter()
            .map(|row| {
                row.iter()
                    .map(|col| col / (1. - (beta1.powf(count))))
                    .collect()
            })
            .collect();
        let bias_gradient_mtt: Vec<f64> = bias_gradient_mt
            .iter()
            .map(|row| row / (1. - (beta1.powf(count + 1.))))
            .collect();

        // s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        let gradient_vtt: Vec<Vec<f64>> = gradient_vt
            .iter()
            .map(|row| {
                row.iter()
                    .map(|col| col / (1. - (beta1.powf(count))))
                    .collect()
            })
            .collect();
        let bias_gradient_vtt: Vec<f64> = bias_gradient_vt
            .iter()
            .map(|row| row / (1. - (beta1.powf(count + 1.))))
            .collect();

        // v_bias_corr / (s_bias_corr.sqrt() + eps)
        let gradients: Vec<Vec<f64>> = gradient_mtt
            .iter()
            .zip(gradient_vtt.iter())
            .map(|(mtt_row, vtt_row)| {
                mtt_row
                    .iter()
                    .zip(vtt_row.iter())
                    .map(|(mtt_col, vtt_col)| mtt_col / (vtt_col.sqrt() + eps))
                    .collect()
            })
            .collect();
        let bias_gradients: Vec<f64> = bias_gradient_mtt
            .iter()
            .zip(bias_gradient_vtt.iter())
            .map(|(mtt_row, vtt_row)| mtt_row / (vtt_row.sqrt() + eps))
            .collect();

        log::debug!(
            "before weights: {:.3?}, gradient: {:.3?}",
            &weights,
            &gradients
        );
        weights
            .iter_mut()
            .zip(gradients.iter())
            .for_each(|(ws, gs)| {
                ws.iter_mut()
                    .zip(gs.iter())
                    .for_each(|(w, g)| *w -= self.learning_rate * g)
            });
        log::debug!("after weights: {:.3?}", &weights);
        log::debug!(
            "before bias: {:.3?}, gradient: {:.3?}",
            &bias,
            &bias_gradients
        );
        bias.iter_mut()
            .zip(bias_gradients.iter())
            .for_each(|(bias, gradient)| *bias -= self.learning_rate * gradient);
        log::debug!("after bias: {:.3?}", &bias);
    }
}
