use std::marker::PhantomData;

use crate::activators::{Activator, Linear};
use crate::layers::Layer;
use crate::network::Network;
use crate::objectives::Objective;
use crate::optimizers::Optimizer;

pub struct NetworkBuilder;

impl NetworkBuilder {
    pub fn new() -> NetworkBuilder {
        NetworkBuilder
    }

    pub fn input(self, input_dim: usize) -> NetworkBuilderWithInput {
        NetworkBuilderWithInput {
            input_dim,
            layers: vec![],
        }
    }
}

pub struct NetworkBuilderWithInput {
    input_dim: usize,
    layers: Vec<Box<Layer>>,
}

impl NetworkBuilderWithInput {
    pub fn add_layer(
        mut self,
        num_nodes: usize,
        activator: Box<dyn Activator>,
    ) -> NetworkBuilderWithInput {
        let layer = Layer::new(self.input_dim, num_nodes, activator, None, None);
        self.layers.push(Box::new(layer));

        NetworkBuilderWithInput {
            // current num_nodes as next layer's input_dim
            input_dim: num_nodes,
            layers: self.layers,
        }
    }

    pub fn add_layer_with_weights_and_bias(
        mut self,
        num_nodes: usize,
        activator: Box<dyn Activator>,
        seed_weights: Vec<Vec<f64>>,
        seed_bias: Vec<f64>,
    ) -> NetworkBuilderWithInput {
        let layer = Layer::new(
            self.input_dim,
            num_nodes,
            activator,
            Some(seed_weights),
            Some(seed_bias),
        );
        self.layers.push(Box::new(layer));

        NetworkBuilderWithInput {
            // current num_nodes as next layer's input_dim
            input_dim: num_nodes,
            layers: self.layers,
        }
    }

    pub fn output(mut self, num_nodes: usize) -> NetworkBuilderWithOutput {
        let layer = Layer::new(self.input_dim, num_nodes, Box::new(Linear), None, None);
        self.layers.push(Box::new(layer));
        NetworkBuilderWithOutput {
            layers: self.layers,
        }
    }

    pub fn output_with_weights_and_bias(
        mut self,
        num_nodes: usize,
        seed_weights: Vec<Vec<f64>>,
        seed_bias: Vec<f64>,
    ) -> NetworkBuilderWithOutput {
        let layer = Layer::new(
            self.input_dim,
            num_nodes,
            Box::new(Linear),
            Some(seed_weights),
            Some(seed_bias),
        );
        self.layers.push(Box::new(layer));
        NetworkBuilderWithOutput {
            layers: self.layers,
        }
    }
}

pub struct NetworkBuilderWithOutput {
    layers: Vec<Box<Layer>>,
}

impl NetworkBuilderWithOutput {
    pub fn minimize_to<A: Activator + 'static, Obj: Objective<A>>(
        self,
        objective: Obj,
    ) -> NetworkBuilderWithObjective<A, Obj> {
        NetworkBuilderWithObjective {
            layers: self.layers,
            objective,
            _marker: PhantomData,
        }
    }
}

pub struct NetworkBuilderWithObjective<A: Activator, Obj: Objective<A>> {
    layers: Vec<Box<Layer>>,
    objective: Obj,
    _marker: PhantomData<A>,
}

impl<A: Activator, Obj: Objective<A>> NetworkBuilderWithObjective<A, Obj> {
    pub fn optimize_with<Opt: Optimizer>(
        self,
        optimizer: Opt,
    ) -> NetworkBuilderWithOptimizer<A, Obj, Opt> {
        NetworkBuilderWithOptimizer {
            layers: self.layers,
            objective: self.objective,
            optimizer,
            _marker: PhantomData,
        }
    }
}

pub struct NetworkBuilderWithOptimizer<A: Activator, Obj: Objective<A>, Opt: Optimizer> {
    layers: Vec<Box<Layer>>,
    objective: Obj,
    optimizer: Opt,
    _marker: PhantomData<A>,
}

impl<A: Activator, Obj: Objective<A>, Opt: Optimizer> NetworkBuilderWithOptimizer<A, Obj, Opt> {
    pub fn build(self) -> Network<A, Obj, Opt> {
        Network::new(self.layers, self.objective, self.optimizer)
    }
}
