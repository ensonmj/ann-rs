use rand::{thread_rng, Rng};

use crate::activators::Activator;

#[derive(Debug)]
pub struct Layer {
    num_nodes: usize, // include bias
    pub bias: f64,    // the weight of bias, assume bias always be 1.
    pub weights: Vec<Vec<f64>>,
    pub activator: Box<dyn Activator>,
}

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
        seed_bias: Option<f64>,
        seed_weights: Option<Vec<Vec<f64>>>,
        activator: Box<dyn Activator>,
    ) -> Self {
        let weights = seed_weights.unwrap_or_else(|| xavier_init(num_inputs, num_nodes));
        assert_eq!(weights.len(), num_nodes);
        for input_weights in &weights {
            assert_eq!(input_weights.len(), num_inputs);
        }

        Layer {
            num_nodes,
            bias: seed_bias.unwrap_or_else(|| thread_rng().gen()),
            weights,
            activator,
        }
    }

    // calculates the output vector with activations
    pub fn calculate_output(&self, inputs: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .map(|input_weights| {
                self.activator.activate(
                    inputs
                        .iter()
                        .enumerate()
                        .map(|(i, input)| input_weights[i] * input)
                        .sum::<f64>()
                        + self.bias,
                )
            })
            .collect()
    }
}
