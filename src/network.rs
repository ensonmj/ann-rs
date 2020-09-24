use rand::seq::SliceRandom;
use rand::thread_rng;
use std::marker::PhantomData;
use textplots::{Chart, Plot, Shape};

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
    pub fn fit(
        &mut self,
        inputs: Vec<Vec<f64>>,
        expected: Vec<Vec<f64>>,
        epochs: usize,
        batch_size: usize,
    ) -> Vec<f64> {
        let mut all_batch_mean_loss = vec![];
        let mut train_pairs: Vec<(Vec<f64>, Vec<f64>)> =
            inputs.into_iter().zip(expected.into_iter()).collect();
        for i in 0..epochs {
            // for train data and labels shuffle
            train_pairs.shuffle(&mut thread_rng());
            train_pairs.chunks(batch_size).enumerate().fold(
                (0, 0, 0.),
                |(total_hit, total_miss, total_loss), (j, train_pairs)| {
                    log::debug!("{:?}", &train_pairs);
                    let (hit, miss, loss) = self.fit_one_batch(train_pairs);

                    let num_pairs = hit + miss;
                    let total_num = (total_hit + total_miss + num_pairs) as f64;

                    let batch_mean_loss = loss / num_pairs as f64;
                    all_batch_mean_loss.push(batch_mean_loss);

                    log::info!(
                        "epoch:[{}, acc:{:.3}, loss:{:.3}], batch:[{}-{}, acc:{:.3} loss:{:.3}]",
                        i,
                        (total_hit + hit) as f64 / total_num as f64,
                        (total_loss + loss) / total_num as f64,
                        j as u64 * num_pairs,
                        (j + 1) as u64 * num_pairs - 1,
                        hit as f64 / num_pairs as f64,
                        batch_mean_loss,
                    );
                    (total_hit + hit, total_miss + miss, total_loss + loss)
                },
            );
        }

        println!("Loss:");
        let losses: Vec<(f32, f32)> = all_batch_mean_loss
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f32, v as f32))
            .collect();
        let xmax = all_batch_mean_loss.len() as f32 / 100.;
        let xmax = if xmax < 20. { 20. } else { xmax };
        let xmax = if xmax > 50. { 50. } else { xmax };
        Chart::new(140, 80, 0., xmax)
            .lineplot(&Shape::Lines(&losses))
            .nice();

        all_batch_mean_loss
    }

    // (batch_hit, batch_miss, batch_loss)
    fn fit_one_batch(&mut self, train_pairs: &[(Vec<f64>, Vec<f64>)]) -> (u64, u64, f64) {
        train_pairs.iter().fold(
            (0, 0, 0.),
            |(batch_hit, batch_miss, batch_loss), (input, expected)| {
                debug_assert_eq!(input.len(), self.layers[0].weights[0].len());
                debug_assert_eq!(expected.len(), self.layers.last().unwrap().weights.len());

                // feed-forward
                // calculate the outputs of each layer in order
                let outputs = self.forward(&input);

                // back propagation
                let gradients = self.backward(&outputs, expected);

                // optimize
                let optimizer = &mut self.optimizer;
                self.layers.iter_mut().zip(gradients.iter()).for_each(
                    |(ref mut layer, (gradient, bias_gradient))| {
                        let weights = &mut layer.weights;
                        let bias = &mut layer.bias;
                        optimizer.optimize(weights, bias, gradient, bias_gradient);
                    },
                );

                let loss = self.objective.loss(&outputs.last().unwrap(), expected);
                let outputs = self.objective.predict_from_logits(&outputs.last().unwrap());
                let hit = outputs
                    .iter()
                    .zip(expected.iter())
                    .fold(true, |eq, (l, r)| eq && if l == r { true } else { false });
                if hit {
                    (batch_hit + 1, batch_miss, batch_loss + loss)
                } else {
                    (batch_hit, batch_miss + 1, batch_loss + loss)
                }
            },
        )
    }

    // infer with pre-trained weights
    pub fn infer(&mut self, input: &[f64]) -> Vec<f64> {
        let outputs = self.forward(input);
        self.objective.predict_from_logits(&outputs.last().unwrap())
    }

    // calc the outputs of each layer in order
    // put the input first in the outputs
    fn forward(&mut self, input: &[f64]) -> Vec<Vec<f64>> {
        debug_assert_eq!(input.len(), self.layers[0].weights[0].len());

        // feed-forward
        // calculate the outputs of each layer in order and find our final answer
        let mut network_outputs = vec![Vec::from(input)];
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
