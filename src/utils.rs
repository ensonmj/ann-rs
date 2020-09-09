pub fn to_onehot(idx: u8) -> [f64; 10] {
    let mut arr = [0.; 10];
    arr[usize::from(idx)] = 1.;
    arr
}

pub fn argmax(arr: &[f64]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

pub fn to_softmax(arr: &[f64]) -> Vec<f64> {
    let exp: Vec<f64> = arr.iter().map(|x| x.exp()).collect();
    let sum_exp: f64 = exp.iter().sum();
    exp.into_iter().map(|v| v / sum_exp).collect()
}
