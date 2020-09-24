use super::Activator;

#[derive(Debug)]
pub struct Relu;

impl Activator for Relu {
    // logits: minibatch of logits from current layer
    // return: minibatch of outputs from current layer
    fn activate(&self, logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
        logits
            .iter()
            .map(|x| x.iter().map(|&x| if x < 0. { 0. } else { x }).collect())
            .collect()
    }

    // outputs: minibatch of outputs of current layer
    // return: minibatch of derivs of current layer
    fn derived(&self, outputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        outputs
            .iter()
            .map(|x| x.iter().map(|&x| if x < 0. { 0. } else { 1. }).collect())
            .collect()
    }
}
