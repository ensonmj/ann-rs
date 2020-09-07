mod elu;
mod relu;
mod sigmoid;

pub use elu::Elu;
pub use relu::Relu;
pub use sigmoid::Sigmoid;

use std::fmt::Debug;

// different activators can be used to train neural networks.
// They all share the same API so they can be defined as a trait!
pub trait Activator: Debug {
    fn activate(&self, v: f64) -> f64;
    fn derived(&self, v: f64) -> f64;
}
