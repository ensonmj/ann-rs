use std::marker::PhantomData;

use crate::activators::Activator;
use crate::layers::Layer;
use crate::objectives::Objective;
use crate::optimizers::Optimizer;

pub struct Network<A: Activator, Obj: Objective<A>, Opt: Optimizer> {
    layers: Vec<Box<Layer>>,
    objective: Obj,
    optimizer: Opt,
    _marker: PhantomData<A>,
}

impl<A: Activator, Obj: Objective<A>, Opt: Optimizer> Network<A, Obj, Opt> {
    pub fn new(layers: Vec<Box<Layer>>, objective: Obj, optimizer: Opt) -> Self {
        Network {
            layers,
            objective,
            optimizer,
            _marker: PhantomData,
        }
    }

    // fit the network, adjust all weights within the network to account for
    // the way that the error after an Example propogates with the weights.
    // return the error value BEFORE this round of training.
    pub fn fit(&mut self, inputs: &[f64], expected: &[f64]) -> f64 {
        debug_assert_eq!(inputs.len(), self.layers[0].weights[0].len());
        debug_assert_eq!(expected.len(), self.layers.last().unwrap().weights.len());
        log::debug!("expected: {:?}", &expected);

        // feed-forward
        // calculate the outputs of each layer in order
        let outputs = self.forward(&inputs);

        // back propagation
        let gradients = self.backward(&outputs, expected);

        // optimize
        let optimizer = &self.optimizer;
        self.layers.iter_mut().zip(gradients.iter()).for_each(
            |(ref mut layer, (gradient, bias_gradient))| {
                let weights = &mut layer.weights;
                let bias = &mut layer.bias;
                optimizer.optimize(weights, bias, gradient, bias_gradient);
            },
        );

        // loss
        self.objective.loss(&outputs.last().unwrap(), expected)
    }

    // infer with pre-trained weights
    pub fn infer(&mut self, inputs: &[f64]) -> f64 {
        let network_outputs = self.forward(inputs);
        self.objective
            .predict_from_probs(&network_outputs.last().unwrap())
    }

    // calc the outputs of each layer in order
    // put the input first in the outputs
    fn forward(&mut self, inputs: &[f64]) -> Vec<Vec<f64>> {
        debug_assert_eq!(inputs.len(), self.layers[0].weights[0].len());

        // feed-forward
        // calculate the outputs of each layer in order and find our final answer
        let mut network_outputs = vec![Vec::from(inputs)];
        for layer in &self.layers {
            let next_output = layer.calc_output(network_outputs.last().unwrap());
            network_outputs.push(next_output);
        }

        network_outputs
    }

    fn backward(
        &mut self,
        outputs: &[Vec<f64>],
        expected: &[f64],
    ) -> Vec<(Vec<Vec<f64>>, Vec<f64>)> {
        let mut gradients: Vec<(Vec<Vec<f64>>, Vec<f64>)> = vec![];
        let mut delta_without_deriv = self
            .objective
            .delta_without_deriv(&outputs.last().unwrap(), expected);

        // loop through the layers backwards and propagate the error throughout
        let mut layers_backwards: Vec<&Layer> = self.layers.iter().map(AsRef::as_ref).collect();
        layers_backwards.reverse();

        // necessary to get around borrow rules
        let num_layers = layers_backwards.len();

        // for layer
        for (k, layer) in layers_backwards.iter().enumerate() {
            let layer_k = num_layers - k;

            let (delta, gradient, bias_gradient) = layer.delta_without_deriv_and_gradient(
                &delta_without_deriv,
                &outputs[layer_k],
                &outputs[layer_k - 1],
            );
            gradients.push((gradient, bias_gradient));

            delta_without_deriv = delta;
        }
        gradients.reverse();
        gradients
    }
}
