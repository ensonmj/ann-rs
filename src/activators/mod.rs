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
    fn activate(&self, x: &[f64]) -> Vec<f64>;
    fn derived(&self, x: &[f64]) -> Vec<f64>;
}
