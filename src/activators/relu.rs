use super::Activator;

#[derive(Debug)]
pub struct Relu;

impl Activator for Relu {
    fn activate(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| if x < 0. { 0. } else { x }).collect()
    }

    fn derived(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| if x < 0. { 0. } else { 1. }).collect()
    }
}
