use crate::activators::Activator;
use crate::functions::he_init;

#[derive(Debug)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>, // the weight of bias, assume bias always be 1.
    pub activator: Box<dyn Activator>,
}

impl Layer {
    // Used as a public API for construction and validation of layers in a network
    // when `None` is specified, the most common defaults are used
    // IE He initialization for seed weights, 0 for bias
    pub fn new(
        input_dim: usize,
        num_nodes: usize,
        activator: Box<dyn Activator>,
        seed_weights: Option<Vec<Vec<f64>>>,
        seed_bias: Option<Vec<f64>>,
    ) -> Self {
        let weights = seed_weights.unwrap_or_else(|| he_init(input_dim, num_nodes));
        assert_eq!(weights.len(), num_nodes);
        for input_weights in &weights {
            assert_eq!(input_weights.len(), input_dim);
        }

        // https://stackoverflow.com/questions/44883861/initial-bias-values-for-a-neural-network
        let bias = seed_bias.unwrap_or(vec![0.; num_nodes]);
        assert_eq!(bias.len(), num_nodes);

        Layer {
            bias,
            weights,
            activator,
        }
    }

    // calculates the output[f(w*x+b)] vector with activations of mini batch
    //
    // inputs: minibatch<layer nodes>
    // return: minibatch<layer nodes>
    pub fn calc_output(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.activator.activate(
            &(inputs
                .iter()
                .map(|input| {
                    self.weights
                        .iter()
                        .zip(self.bias.iter())
                        .map(|(input_weights, bias)| {
                            // calc f(w*x+b) for each node
                            input_weights
                                .iter()
                                .zip(input.iter())
                                .map(|(w, x)| w * x)
                                .sum::<f64>()
                                + bias
                        })
                        .collect()
                })
                .collect::<Vec<Vec<_>>>()),
        )
    }

    // delta rule
    // https://blog.yani.io/deltarule/
    // https://blog.yani.io/backpropagation/
    //
    // delta_without_deriv (without multify prev layer activator's deriv) for previous layer
    // gradient and bias_gradient for current layer
    //
    // curr_delta_without_deriv: minibatch of current layer's delta_without_deriv
    // curr_output: minibatch of current layer's output
    // prev_output: minibatch of previous layer's output
    // return: minibatch of previous layer's delta_without_deriv and current layer's gradients
    pub fn delta_without_deriv_and_gradient(
        &self,
        curr_delta_without_derivs: &[Vec<f64>],
        curr_outputs: &[Vec<f64>],
        prev_outputs: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let curr_deltas = self.delta(curr_delta_without_derivs, curr_outputs);
        let (gradients, bias_gradients) = self.gradient(&curr_deltas, prev_outputs);
        let prev_delta_without_derivs = self.prev_delta_without_deriv(&curr_deltas);

        (prev_delta_without_derivs, gradients, bias_gradients)
    }

    // curr_delta_without_deriv: minibatch of current layer delta_without_deriv
    // curr_output: minibatch of current layer output
    // return: minibatch of current layer's delta
    fn delta(
        &self,
        curr_delta_without_derivs: &[Vec<f64>],
        curr_outputs: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        // curr_delta = curr_delta_without_deriv * deriv
        let derivs = self.activator.derived(curr_outputs);
        curr_delta_without_derivs
            .iter()
            .zip(derivs.iter())
            .map(|(delta, deriv)| {
                delta
                    .iter()
                    .zip(deriv)
                    .map(|(delta, deriv)| delta * deriv)
                    .collect()
            })
            .collect()
    }

    // curr_delta: minibatch of current layer delta
    // prev_output: minibatch of previous layer output
    // return: minibatch of current layer gradients
    fn gradient(
        &self,
        curr_deltas: &[Vec<f64>],
        prev_outputs: &[Vec<f64>],
    ) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        // gradient = curr_delta * prev_output
        let gradients: Vec<Vec<Vec<f64>>> = curr_deltas
            .iter()
            .zip(prev_outputs.iter())
            .map(|(curr_delta, prev_output)| {
                self.weights
                    .iter()
                    .zip(curr_delta.iter())
                    .map(|(input_weights, delta)| {
                        input_weights
                            .iter()
                            .zip(prev_output)
                            .map(|(_, prev_output)| delta * prev_output)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // bias gradient = curr_delta * prev_output = curr_delta * 1
        let bias_gradients = curr_deltas.iter().cloned().collect();

        (gradients, bias_gradients)
    }

    // curr_delta: minibatch of current layer delta
    // return: minibatch of previous layer delta_without_deriv
    fn prev_delta_without_deriv(&self, curr_deltas: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // prev_delta_without_deriv = SUM(curr_delta[i] * weights[j][i]) over j
        let input_dim = self.weights[0].len();
        curr_deltas
            .iter()
            .map(|curr_delta| {
                (0..input_dim)
                    .map(|i| {
                        curr_delta
                            .iter()
                            .enumerate()
                            .map(|(j, delta)| delta * self.weights[j][i])
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}
