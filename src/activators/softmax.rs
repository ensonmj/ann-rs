use super::Activator;

use std::cmp::Ordering;

#[derive(Debug)]
pub struct Softmax;

impl Activator for Softmax {
    // logits: minibatch of logits from current layer
    // return: minibatch of outputs from current layer
    fn activate(&self, logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // softmax(x)=softmax(x+c)
        // use max to overcome overflow or underflow
        logits
            .iter()
            .map(|x| {
                let max = x
                    .iter()
                    .max_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
                    .unwrap();

                let exps: Vec<f64> = x.iter().map(|x| (x - max).exp()).collect();
                let sum_exp: f64 = exps.iter().sum();
                exps.into_iter().map(|v| v / sum_exp).collect()
            })
            .collect()
    }

    // http://blog.prince2015.club/2020/03/27/softmax/
    //
    // j==i: f' = f(j)*(1-f(j))
    // j!=i: f' = -f(j)*f(i)
    // if expecteds[j] == 1
    // for i: 0-n
    //     if i == j: Sj*(1-Sj)
    //     if i != j: -Sj*Si = Sj*(0-Si)
    //
    // outputs: minibatch of outputs of current layer
    // return: minibatch of derivs of current layer
    fn derived(&self, _outputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        unimplemented!()
        // let s = x[node_idx];
        // x.iter()
        //     .enumerate()
        //     .map(|(i, &x)| if i == node_idx { s * (1. - s) } else { -x * s })
        //     .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let x = [vec![1., 2., 3.]];
        let result = Softmax.activate(&x);
        assert_eq!(
            result,
            [[0.09003057317038046, 0.24472847105479764, 0.6652409557748218]]
        );

        let x = [vec![1000., 2000., 3000.]];
        let result = Softmax.activate(&x);
        assert_eq!(result, [[0.0, 0.0, 1.0]]);
    }
}
