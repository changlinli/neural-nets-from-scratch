# %%

print("Hello world")

def assert_with_expect(expected, actual):
    assert expected == actual, f"Expected: {expected} Actual: {actual}"


# %%

def relu(x: float) -> float:
    """
    ReLU (rectified linear unit), one of the simplest non-linear activation
    functions out there.
    """
    # TODO: Fill this in!
    return max(x, 0.0)

assert relu(5.0)

# %%
from dataclasses import dataclass

@dataclass
class Neuron:
    weights: list[float]
    bias: float

    def compute_output(self, inputs: list[float]) -> float:
        """
        Compute what the output of a single neuron should look like.
        """
        assert len(inputs) == len(self.weights)
        # TODO: Fill this in!
        result = 0
        for weight, input in zip(self.weights, inputs):
            result += weight * input
        return relu(result + self.bias)

# Assert

test_neuron = Neuron(weights=[1, 2], bias=1)
assert_with_expect(actual=test_neuron.compute_output([2, 3]), expected=9)
assert_with_expect(actual=test_neuron.compute_output([2, -2]), expected=0)

# %%

def forward_pass_single_layer(input: list[float], layer: list[Neuron]) -> list[float]:
    for neuron in layer:
        assert len(neuron.weights) == len(input)
    # TODO: Fill this in!
    return [neuron.compute_output(input) for neuron in layer]


def forward_pass_network(initial_inputs: list[float], layers: list[list[Neuron]]) -> list[float]:
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
demo_network: list[list[Neuron]] = \
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

# %%

forward_pass_network([0.0, 1.0], demo_network)

# %%

def backpropagation(network: list[list[Neuron]]):
    return None



# %%

import torch as t

# %%

# Here's an example of using PyTorch to automatically calculate a derivative for you.

x = t.tensor([5.0], requires_grad=True)

# Derivative here is 2x + 1, so that should be a derivative of 11 for x = 5
y = x ** 2 + x

# PyTorch's auto-differentiation facilities are based entirely around mutability
# Make sure that you call backward before you look at the gradients!
y.backward()

# x.grad is the numeral calculation of dy/dx at x = 5
print(f"{x.grad=}")

# %%

a = t.tensor([5.0], requires_grad=True)

b = t.tensor([3.0], requires_grad=True)

c = a ** 2 + b ** 2

# Use PyTorch to calculate what dc/da is and what dc/db are.

# TODO: Fill in the Nones!

dc_da = None

assert_with_expect(expected=t.tensor(10.0), actual=dc_da)

dc_db = None

assert_with_expect(expected=t.tensor(6.0), actual=dc_db)

# %%

a_and_b = t.tensor([5.0, 3.0], requires_grad=True)

c = (a_and_b ** 2).sum()

# %%

# Use PyTorch to calculate again what dc/da and what dc/db are

# TODO: Fill in the Nones!

dc_da = None

dc_db = None

# %%

# Return two tensors

import numpy as np

def generate_one_thousand_points() -> t.Tensor:
    return t.rand(1000, 2, requires_grad=True)


assert generate_one_thousand_points().requires_grad


def x_squared_plus_y_squared_plus_5_thousand_times() -> tuple[np.ndarray, np.ndarray]:
    """
    Usually a Tensor will contain both its value and its gradient, here we want
    you to manually separate the two and return them as NumPy arrays (which do
    not have gradients built int).

    We will calculate this for the function f(x, y) = x^2 + y^2 + 5

    So the first element of this tuple should be 

    Note to turn a PyTorch tensor into a NumPy array, call the .detach().numpy() method
    on a tensor.
    """
    points = generate_one_thousand_points()
    result = (points ** 2).sum() + 5
    result.backward()
    return (points.detach().numpy(), points.grad.detach().numpy())

print(f"{x_squared_plus_y_squared_plus_5_thousand_times()=}")


# %%
import einops

def apply_linear_function_to_input(
    neural_net_layer: t.Tensor,
    input_to_layer: t.Tensor,
) -> t.Tensor:
    result = einops.einsum(neural_net_layer, input_to_layer, 'd_output d_input, d_batch d_input -> d_batch d_output')
    return result

example_multiplication = apply_linear_function_to_input(t.rand((1, 2)), t.rand((50, 2)))

# %%

example_parameter = t.rand((1, 2), requires_grad=True)

example_result = apply_linear_function_to_input(example_parameter, t.rand((50, 2)))

# %%

thing = example_result.sum()

# %%

thing.backward()

# %%

example_parameter.grad


# %%

@dataclass
class ThreeLayerNeuralNet:
    layer_0: t.Tensor
    layer_0_bias: t.Tensor
    layer_1: t.Tensor
    layer_1_bias: t.Tensor
    layer_2: t.Tensor
    layer_2_bias: t.Tensor

def initialize_new_three_layer_net() -> ThreeLayerNeuralNet:
    with t.no_grad():
        neural_net = ThreeLayerNeuralNet(
            # We're going to use usual matrix order of dimensions here That is
            # for a matrix mxn, that means we have n-dimensional input and
            # m-dimensional output, so likewise here (300, 96) means
            # 96-dimensional input and 300-dimensional output
            layer_0 = t.zeros((300, 96), requires_grad=True).uniform_(-0.01, 0.01),
            layer_0_bias = t.zeros(300, requires_grad=True).uniform_(-0.01, 0.01),
            layer_1 = t.zeros((400, 300), requires_grad=True).uniform_(-0.01, 0.01),
            layer_1_bias = t.zeros(400, requires_grad=True).uniform_(-0.01, 0.01),
            layer_2 = t.zeros((10, 400), requires_grad=True).uniform_(-0.01, 0.01),
            layer_2_bias = t.zeros(10, requires_grad=True).uniform_(-0.01, 0.01),
        )
        return neural_net


new_neural_net = initialize_new_three_layer_net()

# %%

def sigmoid(x: t.Tensor) -> t.Tensor:
    """
    A sigmoid function. Look at the equation in the documentation for
    https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
    """
    # TODO: Fill this in!
    return 1 / (1 + t.exp(-x))


def forward(x: t.Tensor, neural_net: ThreeLayerNeuralNet) -> t.Tensor:
    # Use t.nn.functional.sigmoid instead of the previous relu to make sure that
    # we have something that works with PyTorch tensors and has both positive
    # and negative values
    after_layer_0 = apply_linear_function_to_input(neural_net.layer_0, x)
    # print(f"{after_layer_0=}")
    after_layer_1 = apply_linear_function_to_input(neural_net.layer_1, after_layer_0)
    # print(f"{after_layer_1=}")
    after_layer_2 = apply_linear_function_to_input(neural_net.layer_2, after_layer_1)
    # print(f"{after_layer_2=}")
    return after_layer_2
    print(f"{apply_linear_function_to_input(neural_net.layer_0, x)[..., :10]=}")
    print(f"{t.nn.functional.sigmoid(apply_linear_function_to_input(neural_net.layer_0, x))[..., :10]=}")
    return apply_linear_function_to_input(neural_net.layer_0, x)[..., :10]

# %%

example_output = forward(neural_net=new_neural_net, x=t.ones((10, 96)))

# %%

example_scalar = example_output.sum()

# %%

example_scalar.backward()

# %%

new_neural_net.layer_0.grad

# %%


def zero_all_gradients(neural_net: ThreeLayerNeuralNet) -> None:
    neural_net.layer_0.grad = None
    neural_net.layer_0_bias.grad = None
    neural_net.layer_1.grad = None
    neural_net.layer_1_bias.grad = None
    neural_net.layer_2.grad = None
    neural_net.layer_2_bias.grad = None


example_0 = forward(t.rand((100, 96)), new_neural_net)
print(f"{example_0.shape=}")


def loss_function(expected_outputs: t.Tensor, actual_outputs: t.Tensor) -> t.Tensor:
    # TODO: Implement this
    # print(f"{expected_outputs.shape=}")
    # print(f"{actual_outputs.shape=}")
    result = ((expected_outputs - actual_outputs) ** 2).mean()
    print(f"{result=}")
    return result



def nudge_tensor_towards_minimum(x: t.Tensor, learning_rate: float) -> None:
    # TODO: Implement this
    with t.no_grad():
        # print(f"{x.grad=}")
        x -= x.grad * learning_rate



def tune_weights_once(
    neural_net: ThreeLayerNeuralNet,
    inputs: t.Tensor, 
    expected_outputs: t.Tensor, 
    learning_rate: float,
) -> None:
    zero_all_gradients(neural_net)
    # TODO: Fill in the rest
    outputs = forward(inputs, neural_net)
    loss = loss_function(
        expected_outputs=expected_outputs, 
        actual_outputs=outputs,
    )
    loss.backward()
    print(f"Internal loss: {loss=}")
    print(f"{neural_net.layer_0.grad=}")
    print(f"{neural_net.layer_2.grad=}")
    # with t.no_grad():
    #     neural_net.layer_0 -= neural_net.layer_0.grad * learning_rate
    # print(f"{neural_net.layer_0=}")
    nudge_tensor_towards_minimum(neural_net.layer_0, learning_rate)
    # nudge_tensor_towards_minimum(neural_net.layer_0_bias, learning_rate)
    nudge_tensor_towards_minimum(neural_net.layer_1, learning_rate)
    # nudge_tensor_towards_minimum(neural_net.layer_1_bias, learning_rate)
    nudge_tensor_towards_minimum(neural_net.layer_2, learning_rate)
    # nudge_tensor_towards_minimum(neural_net.layer_2_bias, learning_rate)

# %%
from tqdm import tqdm

def train(
    neural_net: ThreeLayerNeuralNet,
    inputs: t.Tensor,
    expected_outputs: t.Tensor,
    learning_rate: float,
    number_of_iterations: int,
) -> None:
    print(f"Initial loss was {loss_function(expected_outputs=expected_outputs, actual_outputs=forward(x=inputs, neural_net=neural_net))}")
    for _ in tqdm(range(number_of_iterations)):
        tune_weights_once(neural_net, inputs, expected_outputs, learning_rate)
    print(f"Final loss was {loss_function(expected_outputs=expected_outputs, actual_outputs=forward(x=inputs, neural_net=neural_net))}")

inputs = t.rand((100, 96))
expected_outputs = t.randint(0, 10, (100, 10))

train(
    neural_net=new_neural_net,
    inputs=inputs,
    expected_outputs=expected_outputs,
    learning_rate=0.01,
    number_of_iterations=100,
)

# %%

print(f"{new_neural_net=}")

# %%

