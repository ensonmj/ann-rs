use rand::seq::SliceRandom;
use rand::thread_rng;
use std::marker::PhantomData;
use textplots::{Chart, Plot, Shape};

use crate::activators::Activator;
use crate::functions::{transform_matrix, transform_vec};
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
        expecteds: Vec<Vec<f64>>,
        epochs: usize,
        batch_size: usize,
    ) -> Vec<f64> {
        debug_assert_eq!(inputs[0].len(), self.layers[0].weights[0].len());
        debug_assert_eq!(
            expecteds[0].len(),
            self.layers.last().unwrap().weights.len()
        );
        let mut all_batch_mean_loss = vec![];
        let mut train_pairs: Vec<(Vec<f64>, Vec<f64>)> =
            inputs.into_iter().zip(expecteds.into_iter()).collect();
        for i in 0..epochs {
            // for train data and labels shuffle
            train_pairs.shuffle(&mut thread_rng());
            train_pairs.chunks(batch_size).enumerate().fold(
                (0, 0, 0.),
                |(total_hit, total_miss, total_loss), (j, train_pairs)| {
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
                        j * batch_size,
                        j * batch_size + num_pairs - 1,
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
        let xmax = all_batch_mean_loss.len() as f32;
        Chart::new(180, 100, 0., xmax)
            .lineplot(&Shape::Lines(&losses))
            .nice();

        all_batch_mean_loss
    }

    // (batch_hit, batch_miss, batch_loss)
    fn fit_one_batch(&mut self, train_pairs: &[(Vec<f64>, Vec<f64>)]) -> (usize, usize, f64) {
        let num_of_minibatch = train_pairs.len() as f64;

        // step1. split inputs and ecpecteds from train_pairs and collect separately
        let mut inputs = vec![];
        let mut expecteds = vec![];
        train_pairs.into_iter().for_each(|(input, expected)| {
            inputs.push(input.clone());
            expecteds.push(expected.clone());
        });

        // step2. feed-forward
        // calculate the outputs of each layer in order
        let outputs = self.forward(&inputs);

        // step3. back propagation
        let all_layer_minibatch_gradients = self.backward(&outputs, &expecteds);

        // step4. optimize
        // use mean of minibatch's gradients currently
        // Vec1<(Vec2<Vec3<Vec4<f64>>>, Vec2<Vec3<f64>>)>
        // =>
        // Vec1<(Mean<Vec3<Vec4<f64>>>, Mean<Vec3<f64>>)>
        let mut mean_gradients: Vec<(Vec<Vec<f64>>, Vec<f64>)> = all_layer_minibatch_gradients
            .iter()
            .map(|(batch_gradients, batch_bias_gradients)| {
                let num_nodes = batch_gradients[0].len();
                let input_dim = batch_gradients[0][0].len();
                let mut mean_gradients: Vec<Vec<f64>> = vec![vec![0.; input_dim]; num_nodes];
                batch_gradients.iter().for_each(|gradients| {
                    transform_matrix(&mut mean_gradients, gradients, |sum_g, g| {
                        *sum_g = *sum_g + g
                    })
                });
                mean_gradients.iter_mut().for_each(|row| {
                    row.iter_mut()
                        .for_each(|col| *col = *col / num_of_minibatch)
                });

                let mut mean_bias_gradients: Vec<f64> = vec![0.; num_nodes];
                batch_bias_gradients.iter().for_each(|bias_gradients| {
                    transform_vec(
                        &mut mean_bias_gradients,
                        bias_gradients,
                        |sum_bias_g, bias_g| *sum_bias_g = *sum_bias_g + bias_g,
                    )
                });
                mean_bias_gradients
                    .iter_mut()
                    .for_each(|row| *row = *row / num_of_minibatch);

                (mean_gradients, mean_bias_gradients)
            })
            .collect();

        let optimizer = &mut self.optimizer;
        self.layers
            .iter_mut()
            .zip(mean_gradients.iter_mut())
            .enumerate()
            .for_each(|(idx, (ref mut layer, (gradient, bias_gradient)))| {
                let weights = &mut layer.weights;
                let bias = &mut layer.bias;
                optimizer.optimize(idx, weights, bias, gradient, bias_gradient);
            });

        // step5. evaluation
        // hit_count, miss_count, loss
        let loss = self
            .objective
            .loss(&outputs.last().unwrap(), &expecteds)
            .iter()
            .sum();
        let outputs = self.objective.predict_from_logits(&outputs.last().unwrap());
        let (hit_count, miss_count) = outputs.iter().zip(expecteds.iter()).fold(
            (0, 0),
            |(hit_count, miss_count), (output, expected)| {
                let hit = output
                    .iter()
                    .zip(expected)
                    .fold(true, |eq, (l, r)| eq && if l == r { true } else { false });
                if hit {
                    (hit_count + 1, miss_count)
                } else {
                    (hit_count, miss_count + 1)
                }
            },
        );
        (hit_count, miss_count, loss)
    }

    // infer with pre-trained weights
    pub fn infer(&mut self, input: &[f64]) -> Vec<f64> {
        let outputs = self.forward(&[Vec::from(input)]);
        self.objective
            .predict_from_logits(&outputs.last().unwrap())
            .first()
            .unwrap()
            .clone()
    }

    // calc the outputs of each layer in order
    // put the input first in the outputs
    // inputs: minibatch of Vec<f64>
    // return: all layers' of minibatch outputs include inputs
    fn forward(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<Vec<f64>>> {
        // layers<minibatch<layer nodes>>`
        let mut network_outputs = vec![Vec::from(inputs)];
        // calculate the outputs of each layer in order and find our final answer
        for layer in &self.layers {
            let next_output = layer.calc_output(network_outputs.last().unwrap());
            network_outputs.push(next_output);
        }

        network_outputs
    }

    // outputs: all layers' of minibatch outputs include inputs
    // expected: minibatch of labels
    // return: all layers' of (minibatch gradients, minibatch bias_gradients)
    fn backward(
        &mut self,
        outputs: &[Vec<Vec<f64>>],
        expecteds: &[Vec<f64>],
    ) -> Vec<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>)> {
        let mut all_layer_gradients: Vec<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>)> = vec![];
        let mut delta_without_derivs = self
            .objective
            .delta_without_deriv(&outputs.last().unwrap(), expecteds);

        // loop through the layers backwards and propagate the error throughout
        let mut layers_backwards: Vec<&Layer> = self.layers.iter().map(AsRef::as_ref).collect();
        layers_backwards.reverse();

        // necessary to get around borrow rules
        let num_layers = layers_backwards.len();

        // for layer
        for (k, layer) in layers_backwards.iter().enumerate() {
            let layer_k = num_layers - k;

            let (deltas, gradients, bias_gradients) = layer.delta_without_deriv_and_gradient(
                &delta_without_derivs,
                &outputs[layer_k],
                &outputs[layer_k - 1],
            );
            all_layer_gradients.push((gradients, bias_gradients));

            delta_without_derivs = deltas;
        }
        all_layer_gradients.reverse();
        all_layer_gradients
    }
}
