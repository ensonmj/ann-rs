use super::Activator;
use crate::functions::sigmoid;

#[derive(Debug)]
pub struct Sigmoid;

impl Activator for Sigmoid {
    // logits: minibatch of logits from current layer
    // return: minibatch of outputs from current layer
    fn activate(&self, logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
        logits
            .iter()
            .map(|x| x.iter().map(|&x| sigmoid(x)).collect())
            .collect()
    }

    // f'(x)=f(x)(1-f(x))
    //
    // outputs: minibatch of outputs of current layer
    // return: minibatch of derivs of current layer
    fn derived(&self, outputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        outputs
            .iter()
            .map(|x| x.iter().map(|&x| sigmoid(x) * (1. - sigmoid(x))).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let x = [vec![3.5]];
        let result = Sigmoid.activate(&x);
        assert_eq!(result, [[0.9706877692486436]]);
    }
}
