mod elu;
mod linear;
mod relu;
mod sigmoid;
mod softmax;

pub use elu::Elu;
pub use linear::Linear;
pub use relu::Relu;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;

use std::fmt::Debug;

// different activators can be used to train neural networks.
// They all share the same API so they can be defined as a trait!
pub trait Activator: Debug {
    // logits: minibatch of logits from current layer
    // return: minibatch of outputs from current layer
    fn activate(&self, logits: &[Vec<f64>]) -> Vec<Vec<f64>>;
    // outputs: minibatch of outputs of current layer
    // return: minibatch of derivs of current layer
    fn derived(&self, outputs: &[Vec<f64>]) -> Vec<Vec<f64>>;
}
