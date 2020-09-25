use pretty_env_logger;

use ann_rs::activators::Sigmoid;
// use ann_rs::functions::xavier_init;
use ann_rs::objectives::BinaryCrossEntropy;
use ann_rs::optimizers::Adam;
// use ann_rs::optimizers::SGD;
use ann_rs::NetworkBuilder;
// use rand::{thread_rng, Rng};

fn main() {
    pretty_env_logger::init();

    // case 1 with final weights and bias
    // https://stackoverflow.com/questions/49676471/backpropagation-for-my-own-neural-net-to-solve-xor-not-converging-correctly
    // let weights1 = vec![vec![20., 20.], vec![-20., -20.]]; // OR, NAND
    // let bias1 = vec![-10., 30.];
    // let weights2 = vec![vec![20., 20.]]; // AND
    // let bias2 = vec![-30.];
    // let mut nn = NetworkBuilder::new()
    //     .input(2)
    //     .add_layer_with_weights_and_bias(2, Box::new(Sigmoid), weights1, bias1)
    //     .output_with_weights_and_bias(1, weights2, bias2)
    //     .output(1)
    //     .minimize_to(BinaryCrossEntropy::new())
    //     .optimize_with(SGD::new(0.1, 1.))
    //     .build();

    // case 2 with random weights and bias
    // let weights1 = xavier_init(2, 2)
    //     .iter()
    //     .map(|row| row.iter().map(|v| v * 2.5).collect())
    //     .collect();
    // let bias1 = (0..2).map(|_| thread_rng().gen::<f64>() * 20.).collect();
    // let mut nn = NetworkBuilder::new()
    //     .input(2)
    //     .add_layer_with_weights_and_bias(2, Box::new(Sigmoid), weights1, bias1)
    //     .output(1)
    //     .minimize_to(BinaryCrossEntropy::new())
    //     .optimize_with(SGD::new(0.1, 1.))
    //     .build();

    // case 3
    let mut nn = NetworkBuilder::new()
        .input(2)
        .add_layer(10, Box::new(Sigmoid))
        .output(1)
        .minimize_to(BinaryCrossEntropy::new())
        // .optimize_with(SGD::new(0.8))
        .optimize_with(Adam::new(0.01))
        .build();

    let inputs = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
    let labels = vec![vec![0.], vec![1.], vec![1.], vec![0.]];

    // train 10'000 times
    nn.fit(inputs.clone(), labels.clone(), 2000, 4);
    log::info!("all inputs: {:?}", &inputs);
    log::info!("all labels: {:?}", &labels);
    log::info!(
        "all infers: {:?}",
        [
            nn.infer(&inputs[0]),
            nn.infer(&inputs[1]),
            nn.infer(&inputs[2]),
            nn.infer(&inputs[3])
        ]
    );
}
