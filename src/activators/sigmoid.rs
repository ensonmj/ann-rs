use super::Activator;

#[derive(Debug)]
pub struct Sigmoid;

impl Activator for Sigmoid {
    fn activate(&self, x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }

    fn derived(&self, x: f64) -> f64 {
        x * (1. - x)
    }
}
