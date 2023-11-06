from dataclasses import dataclass


def relu(x: float) -> float:
    if x < 0:
        return 0
    else:
        return x


@dataclass
class Neuron:
    weights: [float]
    bias: float

    def compute_output(self, inputs: [float]) -> float:
        current_sum = 0
        for (input, weight) in zip(inputs, self.weights):
            current_sum += input * weight
        return relu(current_sum + self.bias)


def forward_pass_single_layer(input: [float], layer: [Neuron]) -> [float]:
    return [neuron.compute_output(input) for neuron in layer]


def forward_pass_network(initial_inputs: [float], layers: [[Neuron]]) -> [float]:
    last_output = initial_inputs
    for layer in layers:
        last_output = forward_pass_single_layer(last_output, layer)
    return last_output


# Neural net that takes in three inputs and has two outputs, and has three layers: 3 neurons, 2 neurons, and 2 neurons
# Notice that:
#   1. Because we take in two inputs, the first layer of neurons all have two weights
#   2. Because there are three neurons that feed into the second layer, all the neurons of the second layer have three
#      weights
#   3. Because there are two neurons in the second layer, all the neurons of the third layer have two weights
#   4. We have three inputs and two outputs because the first layer has three neurons and the last layer has two neurons
demo_network: [[Neuron]] = \
    [
        [
            Neuron(weights=[0.1, 0.2], bias=0.3),
            Neuron(weights=[-0.15, 0.1], bias=-0.1),
            Neuron(weights=[0.2, 0.1], bias=0.1),
        ],
        [
            Neuron(weights=[0.1, 0.2, 0.3], bias=0.3),
            Neuron(weights=[-0.15, 0.1, 0.9], bias=-0.1),
        ],
        [
            Neuron(weights=[0.1, 0.2], bias=0.3),
            Neuron(weights=[-0.15, 0.1], bias=-0.1),
        ],
    ]


def main() -> None:
    print(forward_pass_network([0.0, 1.0, 2.0], demo_network))


if __name__ == "__main__":
    main()
