use super::Activator;

#[derive(Debug)]
pub struct Linear;

impl Activator for Linear {
    // logits: minibatch of logits from current layer
    // return: minibatch of outputs from current layer
    fn activate(&self, logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
        logits.to_owned()
    }

    // outputs: minibatch of outputs of current layer
    // return: minibatch of derivs of current layer
    fn derived(&self, outputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        outputs
            .iter()
            .map(|x| x.iter().map(|_| 1.).collect())
            .collect()
    }
}
