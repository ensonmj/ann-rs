use super::Activator;

#[derive(Debug)]
pub struct Linear;

impl Activator for Linear {
    fn activate(&self, x: &[f64]) -> Vec<f64> {
        x.to_owned()
    }

    fn derived(&self, x: &[f64]) -> Vec<f64> {
        vec![1.; x.len()]
    }
}
