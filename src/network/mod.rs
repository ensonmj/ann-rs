mod layer;

pub use layer::Layer;

use crate::activators::{Activator, Relu};

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
}

// Used as a public API for construction and validation of layers in a network
// when `None` is specified, the most common defaults are used
// IE xavier initialization for seed weights, uniform random for bias, Relu for activation.
pub struct LayerBlueprint {
    num_nodes: usize,
    seed_bias: Option<Vec<f64>>,
    seed_weights: Option<Vec<Vec<f64>>>,
    activator: Box<dyn Activator>,
}

impl LayerBlueprint {
    pub fn new(num_nodes: usize) -> Self {
        LayerBlueprint {
            num_nodes,
            seed_bias: None,
            seed_weights: None,
            activator: Box::new(Relu),
        }
    }

    pub fn bias(mut self, bias: Vec<f64>) -> Self {
        self.seed_bias = Some(bias);
        self
    }

    pub fn weights(mut self, weights: Vec<Vec<f64>>) -> Self {
        assert_eq!(self.num_nodes, weights.len());
        self.seed_weights = Some(weights);
        self
    }

    pub fn activator<A: 'static + Activator>(mut self, activator: A) -> Self {
        self.activator = Box::new(activator);
        self
    }
}

impl Network {
    pub fn new(mut num_inputs: usize, layer_blueprints: Vec<LayerBlueprint>) -> Self {
        Network {
            layers: layer_blueprints
                .into_iter()
                .map(|blueprint| {
                    let layer = Layer::new(
                        num_inputs,
                        blueprint.num_nodes,
                        blueprint.seed_weights,
                        blueprint.seed_bias,
                        blueprint.activator,
                    );
                    num_inputs = blueprint.num_nodes;
                    layer
                })
                .collect(),
        }
    }

    // trains the network, adjust all weights within the network to account for
    // the way that the error after an Example propogates with the weights.
    // return the error value BEFORE this round of training.
    pub fn train(
        &mut self,
        inputs: &[f64],
        labels: &[f64],
        lr: f64,
        cost_fn: impl Fn(&[f64], &[f64]) -> Vec<f64>,
    ) -> f64 {
        assert_eq!(inputs.len(), self.layers[0].weights[0].len());
        assert_eq!(labels.len(), self.layers.last().unwrap().weights.len());

        // feed-forward
        // calculate the outputs of each layer in order and find our final answer
        let mut network_outputs = vec![Vec::from(inputs)];
        for layer in &self.layers {
            let next_output = layer.calculate_output(network_outputs.last().unwrap());
            network_outputs.push(next_output);
        }

        // keep track of the error the current layer being adjusted is experiencing.
        let mut layer_error = cost_fn(labels, &network_outputs.last().unwrap());

        // net error for BEFORE the training (to return)
        let net_error = layer_error.iter().map(|x| 0.5 * x.powi(2)).sum();

        // back propagation
        // loop through the layers backwards and propagate the error throughout
        let mut layers_backwards: Vec<&mut Layer> = self.layers.iter_mut().collect();
        layers_backwards.reverse();
        // necessary to get around borrow rules
        let num_layers = layers_backwards.len();
        for (layer_i, layer) in layers_backwards.iter_mut().enumerate() {
            let layer_i = num_layers - layer_i;
            // to build up the next layer's error
            let mut next_layer_error: Vec<f64> = layer.weights[0].iter().map(|_| 0.).collect();

            for (out_i, input_weights) in layer.weights.iter_mut().enumerate() {
                // do the chain rule dance!
                let input_err =
                    layer_error[out_i] * layer.activator.derived(network_outputs[layer_i][out_i]);
                // adjust layer weights
                for (in_i, weight) in input_weights.iter_mut().enumerate() {
                    next_layer_error[in_i] += input_err * *weight;

                    // w(t+1) = w(t) + lr * (expected(t) - predicted(t)) * x(t)
                    *weight += lr * input_err * network_outputs[layer_i - 1][in_i];
                }
                // adjust layer bias as well for faster training
                // bias(t+1) = bias(t) + lr * (expected(t) - predicted(t))
                layer.bias[out_i] += lr * input_err;
            }

            layer_error = next_layer_error;
        }

        net_error
    }

    // infer with pre-trained weights
    pub fn infer(&mut self, inputs: &[f64], outputs: &mut [f64]) {
        assert_eq!(inputs.len(), self.layers[0].weights[0].len());
        assert_eq!(outputs.len(), self.layers.last().unwrap().weights.len());

        // feed-forward
        // calculate the outputs of each layer in order and find our final answer
        let mut network_outputs = vec![Vec::from(inputs)];
        for layer in &self.layers {
            let next_output = layer.calculate_output(network_outputs.last().unwrap());
            network_outputs.push(next_output);
        }

        outputs.copy_from_slice(&network_outputs.last().unwrap()[..])
    }
}
