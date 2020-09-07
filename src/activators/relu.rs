use super::Activator;

#[derive(Debug)]
pub struct Relu;

impl Activator for Relu {
    fn activate(&self, x: f64) -> f64 {
        if x < 0. {
            0.
        } else {
            x
        }
    }

    fn derived(&self, x: f64) -> f64 {
        if x < 0. {
            0.
        } else {
            1.
        }
    }
}
