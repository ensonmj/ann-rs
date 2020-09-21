use super::Activator;

#[derive(Debug)]
pub struct Elu;

impl Activator for Elu {
    fn activate(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&x| if x < 0. { x.exp() - 1. } else { x })
            .collect()
    }

    fn derived(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&x| if x < 0. { x.exp() } else { 1. })
            .collect()
    }
}
