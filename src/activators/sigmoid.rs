use super::Activator;
use crate::functions::sigmoid;

#[derive(Debug)]
pub struct Sigmoid;

impl Activator for Sigmoid {
    fn activate(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| sigmoid(x)).collect()
    }

    // f'(x)=f(x)(1-f(x))
    fn derived(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| sigmoid(x) * (1. - sigmoid(x))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let x = [3.5];
        let result = Sigmoid.activate(&x);
        assert_eq!(result, [0.9706877692486436]);
    }
}
