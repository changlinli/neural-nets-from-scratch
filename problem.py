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

# Here's an example of using PyTorch to automatically calculate a derivative for
# you. When we are manually creating tensors, we have to explicitly tell PyTorch
# to remember we want to calculate the gradient for this tensor, so we should
# pass in requires_grad=True. As we use PyTorch more and more, we'll see a lot
# of library calls that will automatically take care of this for us.
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

# %%

# Use PyTorch to calculate what dc/da is and what dc/db are.

# TODO: Fill in the Nones!
# Remember to first populate the gradients before calling .grad!
c.backward()

dc_da = a.grad

assert_with_expect(expected=t.tensor(10.0), actual=dc_da)

dc_db = b.grad

assert_with_expect(expected=t.tensor(6.0), actual=dc_db)

# %%

a_and_b = t.tensor([5.0, 3.0], requires_grad=True)

c = (a_and_b ** 2).sum()

# %%

# Use PyTorch to calculate again what dc/da and what dc/db are

# TODO: Fill in the Nones!
# Remember to first populate the gradients before calling .grad!
c.backward()

dc_da = a_and_b.grad[0]

dc_db = a_and_b.grad[1]

# %%

# Note that gradients accumulate!

some_input = t.tensor([1.0], requires_grad=True)

some_output = 10 * some_input

some_output.backward()

# Normally because this is just y = 10 * x, we would expect the x's gradient to be 10 at this point

print(f"{some_input.grad=}")

# %%

# PyTorch is smart enough to warn us if we try to use backward again

try:
    some_output.backward()
except RuntimeError as e:
    print(f"PyTorch was smart enough to blow up and prevent us from going backward again with the following message:\n{str(e)}")

# %%

# But PyTorch doesn't warn us if we create a new output reuses `some_input` and
# instead will just keep adding more gradients to the pre-existing gradient.
# This is known as "accumulating gradients," and there are reasons you might
# want to do this, but for our purposes, this is undesirable, as it will give us
# the wrong derivatives/gradients.

another_output = 5 * some_input

another_output.backward()

# Note that we've added two derivatives together, 10 + 5, which is not the
# correct derivative for y = 10 * x or y = 5 * x!
assert some_input.grad == 15

print(f"{some_input.grad=}")

# %%

# Because PyTorch will either throw an error on a given backwards call, or it'll
# accumulate gradients when you potentially don't want that to happen, we
# generally will want to reset gradients between calls to backward(). The
# easiest way to do this is to set `.grad = None`. We'll see later how to do
# this in a less manual fashion.

some_input.grad = None

yet_another_output = 5 * some_input

yet_another_output.backward()

# This time we get the correct gradient!
assert some_input.grad == 5

# %%

# Note that even after resetting a gradient, we still can't call backward again
# on the same output. This has to do with the details of how PyTorch
# automatically calculates derivatives. The exact details of why this is the
# case are irrelevant at the moment (although they may become more apparent when
# we implement backpropagation ourselves), but feel free to ask if you're
# curious.

some_input.grad = None

try:
    yet_another_output.backward()
except RuntimeError as e:
    print(f"Yep we still get the following error message:\n{str(e)}")

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
    """
    Initialize our 
    """
    with t.no_grad():
        neural_net = ThreeLayerNeuralNet(
            # We're going to use usual matrix order of dimensions here That is
            # for a matrix mxn, that means we have n-dimensional input and
            # m-dimensional output, so likewise here (300, 784) means
            # 784-dimensional input and 300-dimensional output
            layer_0 = t.zeros((2000, 784), requires_grad=True).uniform_(-1, 1),
            layer_0_bias = t.zeros(2000, requires_grad=True).uniform_(-1, 1),
            # TODO: Finish implementing 
            layer_1 = t.zeros((400, 2000), requires_grad=True).uniform_(-1, 1),
            layer_1_bias = t.zeros(400, requires_grad=True).uniform_(-1, 1),
            layer_2 = t.zeros((10, 400), requires_grad=True).uniform_(-1, 1),
            layer_2_bias = t.zeros(10, requires_grad=True).uniform_(-1, 1),
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
    after_layer_0 = t.nn.functional.leaky_relu(apply_linear_function_to_input(neural_net.layer_0, x) + neural_net.layer_0_bias)
    # TODO: Fill in the rest of this!
    after_layer_1 = t.nn.functional.leaky_relu(apply_linear_function_to_input(neural_net.layer_1, after_layer_0) + neural_net.layer_1_bias)
    after_layer_2 = t.nn.functional.softmax(apply_linear_function_to_input(neural_net.layer_2, after_layer_1) + neural_net.layer_2_bias, dim=-1)
    return after_layer_2

# %%

example_output = forward(neural_net=new_neural_net, x=t.ones((10, 784)))

# %%

# Note that generally speaking we'll mainly be using scalar (i.e. 0-dimensional
# tensor) outputs that we call .backward() on. This is not too much of a
# limitation because almost always a loss function will output a scalar. There
# are ways to deal with non-scalar outputs, but it's irrelevant to us at the
# moment and for now we'll just point out that trying to do so will cause an
# error.
try:
    # example_output is not a scalar!
    example_output.backward()
except RuntimeError as e:
    print(f"Weren't able to calculate a gradient because of:\n{str(e)}")

# %%

# Notice now that we've turned example_output into a scalar, our call to
# .backward() proceeds with no problem!
example_scalar = example_output.sum()

example_scalar.backward()

# %%

# And we can calculate the gradient of one of our neural net layers relative to
# this scalar!

print(f"{new_neural_net.layer_0=}")
print(f"{new_neural_net.layer_0.grad=}")

# %%

def zero_all_gradients(neural_net: ThreeLayerNeuralNet) -> None:
    neural_net.layer_0.grad = None
    neural_net.layer_0_bias.grad = None
    # TODO: Finish implementing this for all the other layers in our neural net
    neural_net.layer_1.grad = None
    neural_net.layer_1_bias.grad = None
    neural_net.layer_2.grad = None
    neural_net.layer_2_bias.grad = None


def loss_function(expected_outputs: t.Tensor, actual_outputs: t.Tensor) -> t.Tensor:
    # TODO: Implement this
    result = ((expected_outputs - actual_outputs) ** 2).mean()
    return result



def nudge_tensor_towards_minimum(x: t.Tensor, learning_rate: float) -> None:
    with t.no_grad():
        # TODO: Implement this
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
    nudge_tensor_towards_minimum(neural_net.layer_0, learning_rate)
    nudge_tensor_towards_minimum(neural_net.layer_0_bias, learning_rate)
    nudge_tensor_towards_minimum(neural_net.layer_1, learning_rate)
    nudge_tensor_towards_minimum(neural_net.layer_1_bias, learning_rate)
    nudge_tensor_towards_minimum(neural_net.layer_2, learning_rate)
    nudge_tensor_towards_minimum(neural_net.layer_2_bias, learning_rate)

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

# %%

inputs = t.rand((100, 784))
expected_outputs_in_training = t.rand((100, 10))

# %%

inputs[1:2]

# %%

forward(inputs, new_neural_net)

# %%

# train(
#     neural_net=new_neural_net,
#     inputs=inputs,
#     expected_outputs=expected_outputs,
#     # A learning rate of 2 is usually much too high, but we've made some sub-optimal choices in designing our 
#     learning_rate=2,
#     number_of_iterations=1000,
# )


# %%
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# %%

dataset = MNIST(root="data", download=True, transform=ToTensor())

# %%

# %%

img, label = dataset[0]

print(f"This image is meant to express this numeral: {label}")
plt.imshow(img.squeeze())

# %%

def one_hot_encoding(i: int, num_classes: int) -> t.Tensor:
    # TODO: Implement this!
    result = t.zeros([num_classes])
    result[i] = 1
    return result


def make_img_1d(imgs: t.Tensor) -> t.Tensor:
    # TODO: Implement this!
    return einops.rearrange(imgs, '... h w -> ... (h w)')


# %%

# This is an inefficient way of using
training_imgs = []
expected_outputs_in_training = []
non_training_imgs = []
expected_outputs_in_non_training = []
counter = 0
total_imgs = 2000
num_of_training_imgs = 1000
for img, label in dataset:
    if counter >= total_imgs:
        break
    if counter < num_of_training_imgs:
        training_imgs.append(make_img_1d(img).squeeze())
        expected_outputs_in_training.append(one_hot_encoding(label, num_classes=10))
    else:
        non_training_imgs.append(make_img_1d(img).squeeze())
        expected_outputs_in_non_training.append(one_hot_encoding(label, num_classes=10))
    counter += 1

training_imgs = t.stack(training_imgs)
expected_outputs_in_training = t.stack(expected_outputs_in_training)

# %%

print(f"{training_imgs.shape=}")

# %%

train(
    neural_net=new_neural_net,
    inputs=training_imgs,
    expected_outputs=expected_outputs_in_training,
    # A learning rate of 2 is usually much too high, but we've made some sub-optimal choices in designing our 
    learning_rate=1,
    number_of_iterations=10,
)

# %%

expected_outputs_in_training[1]

# %%
print(f"{forward(training_imgs[100:101], new_neural_net)=}")
print(f"{expected_outputs_in_training[100:101]}")

# %%

class SimpleNeuralNet(t.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.implementation = t.nn.Sequential(
            t.nn.Linear(in_features=784, out_features=2000),
            t.nn.ReLU(),
            t.nn.Linear(in_features=2000, out_features=400),
            t.nn.ReLU(),
            t.nn.Linear(in_features=400, out_features=10),
            t.nn.Softmax(dim=-1),
        )

    def forward(self, t: t.Tensor):
        return self.implementation(t)


def train(model: SimpleNeuralNet, epochs: int, lr: int):
    optimizer = t.optim.AdamW(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        output = model(training_imgs)
        # For those who are confused why we use MSE loss here for a
        # classification task, see https://arxiv.org/abs/2006.07322
        loss = t.nn.functional.mse_loss(output, expected_outputs_in_training)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0:
            print(f"Initial loss: {loss=}")
        elif epoch == epochs - 1:
            print(f"Final loss: {loss=}")

model = SimpleNeuralNet()

# %%

train(model, epochs=100, lr=0.001)

# %%

# Let's look at an image that wasn't part of the training data

non_training_img_idx = 0
img_outside_of_training_dataset = non_training_imgs[non_training_img_idx]
label = expected_outputs_in_non_training[non_training_img_idx].argmax()

print(f"Expected label: {label}")
plt.imshow(einops.rearrange(img_outside_of_training_dataset, '(h w) -> h w', h=28))

model_all_guesses = model(img_outside_of_training_dataset)
model_guess_highest_prob = model(img_outside_of_training_dataset).argmax()

print(f"Model guessed this was: {model_guess_highest_prob}")