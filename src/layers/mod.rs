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
    // IE xavier initialization for seed weights, uniform random for bias, Liner for activation.
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

    // calculates the output[f(w*x+b)] vector with activations
    pub fn calc_output(&self, inputs: &[f64]) -> Vec<f64> {
        self.activator.activate(
            &(self
                .weights
                .iter()
                .zip(self.bias.iter())
                .map(|(input_weights, bias)| {
                    // calc f(w*x+b) for each node
                    input_weights
                        .iter()
                        .zip(inputs.iter())
                        .map(|(w, x)| w * x)
                        .sum::<f64>()
                        + bias
                })
                .collect::<Vec<_>>()),
        )
    }

    // delta_without_deriv (without multify prev layer activator's deriv) for prev layer
    // gradient and bias_gradient for curr layer
    pub fn delta_without_deriv_and_gradient(
        &self,
        curr_delta_without_deriv: &[f64],
        curr_output: &[f64],
        prev_output: &[f64],
    ) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        // delta rule
        // https://blog.yani.io/deltarule/
        // https://blog.yani.io/backpropagation/
        let curr_delta = self.delta(curr_delta_without_deriv, curr_output);
        let (gradient, bias_gradient) = self.gradient(&curr_delta, prev_output);
        let prev_delta_without_deriv = self.prev_delta_without_deriv(&curr_delta);

        (prev_delta_without_deriv, gradient, bias_gradient)
    }

    fn delta(&self, curr_delta_without_deriv: &[f64], curr_output: &[f64]) -> Vec<f64> {
        // curr_delta = curr_delta_without_deriv * deriv
        let driv = self.activator.derived(curr_output);
        curr_delta_without_deriv
            .iter()
            .zip(driv.iter())
            .map(|(delta, driv)| delta * driv)
            .collect()
    }

    fn gradient(&self, curr_delta: &[f64], prev_output: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
        // gradient = curr_delta * prev_output
        let gradient: Vec<Vec<f64>> = self
            .weights
            .iter()
            .zip(curr_delta.iter())
            .map(|(input_weights, delta)| {
                input_weights
                    .iter()
                    .zip(prev_output)
                    .map(|(_, prev_output)| delta * prev_output)
                    .collect()
            })
            .collect();
        debug_assert_eq!(self.weights.len(), gradient.len());
        debug_assert_eq!(self.weights[0].len(), gradient[0].len());

        // bias gradient = curr_delta * prev_output = curr_delta * 1
        let bias_gradient = curr_delta.iter().cloned().collect();

        (gradient, bias_gradient)
    }

    fn prev_delta_without_deriv(&self, curr_delta: &[f64]) -> Vec<f64> {
        // prev_delta_without_deriv = SUM(curr_delta[i] * weights[j][i]) over j
        let input_dim = self.weights[0].len();
        (0..input_dim)
            .map(|i| {
                curr_delta
                    .iter()
                    .enumerate()
                    .map(|(j, delta)| delta * self.weights[j][i])
                    .sum()
            })
            .collect()
    }
}
