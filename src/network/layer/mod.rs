use rand::{thread_rng, Rng};

use crate::activators::Activator;

#[derive(Debug)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>, // the weight of bias, assume bias always be 1.
    pub activator: Box<dyn Activator>,
}

// create random weight matrix: vec![vec![float; n_in]; n_out]
fn xavier_init(n_in: usize, n_out: usize) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    let variance = (2f64 / ((n_in + n_out) as f64)).sqrt();
    (0..n_out)
        .map(|_| {
            (0..n_in)
                .map(|_| rng.gen_range(-variance, variance))
                .collect()
        })
        .collect()
}

impl Layer {
    pub fn new(
        num_inputs: usize,
        num_nodes: usize,
        seed_weights: Option<Vec<Vec<f64>>>,
        seed_bias: Option<Vec<f64>>,
        activator: Box<dyn Activator>,
    ) -> Self {
        let weights = seed_weights.unwrap_or_else(|| xavier_init(num_inputs, num_nodes));
        assert_eq!(weights.len(), num_nodes);
        for input_weights in &weights {
            assert_eq!(input_weights.len(), num_inputs);
        }

        let bias =
            seed_bias.unwrap_or_else(|| (0..num_nodes).map(|_| thread_rng().gen()).collect());
        assert_eq!(bias.len(), num_nodes);

        Layer {
            bias,
            weights,
            activator,
        }
    }

    // calculates the output vector with activations
    pub fn calculate_output(&self, inputs: &[f64]) -> Vec<f64> {
        // self.weights
        //     .iter()
        //     .map(|input_weights| {
        //         // calc f(w*x+b) for each node
        //         self.activator.activate(
        //             input_weights
        //                 .iter()
        //                 .zip(inputs.iter())
        //                 .map(|(w, i)| w * i)
        //                 .sum::<f64>()
        //                 + self.bias,
        //         )
        //     })
        //     .collect()
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(input_weights, bias)| {
                // calc f(w*x+b) for each node
                self.activator.activate(
                    input_weights
                        .iter()
                        .zip(inputs.iter())
                        .map(|(w, x)| w * x)
                        .sum::<f64>()
                        + bias,
                )
            })
            .collect()
    }
}
