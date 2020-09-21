use log;
use pretty_env_logger;

use ann_rs::activators::Sigmoid;
use ann_rs::objectives::BinaryCrossEntropy;
use ann_rs::optimizers::SGD;
use ann_rs::NetworkBuilder;

fn main() {
    pretty_env_logger::init();

    // create a network with 3 layers:
    // the first having three nodes using Sigmoid activation and an initial bias of 0.5
    // the next having 10 nodes using Elu activation
    // the final (output) layer having 1 node and a pre-defined seed weight
    let mut nn = NetworkBuilder::new()
        .input(2)
        .add_layer(10, Box::new(Sigmoid))
        .output(1)
        .minimize_to(BinaryCrossEntropy::new())
        .optimize_with(SGD::new(0.1))
        .build();

    let inputs = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
    let labels = vec![0., 1., 1., 0.];

    // train 10'000 times with a variable learning rate
    for i in 0..10000 {
        let mut errors = vec![];
        for (i, input) in inputs.iter().enumerate() {
            let label = &labels[i];
            errors.push(nn.fit(input, &[*label]));
        }

        if i % 100 == 0 {
            let mean_errors = errors.iter().sum::<f64>() / (errors.len() as f64);
            log::info!("{}: {}", i, mean_errors);
        }
    }
    let output = nn.infer(&inputs[2]);
    println!("{:?}", output);
}
