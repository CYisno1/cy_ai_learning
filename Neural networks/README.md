# 🧠 Micrograd: A Tiny Scalar Autograd Engine and Neural Network Library

A from-scratch implementation of a scalar-valued automatic differentiation engine and a simple Multi-Layer Perceptron library, inspired by Andrej Karpathy's **micrograd**.

This project demonstrates the core mechanics behind modern deep learning frameworks, including computational graphs, reverse-mode automatic differentiation, backpropagation, gradient accumulation, and gradient-based optimization.

### Project Structure

```text
micrograd-project/
├── micrograd/
│   ├── __init__.py       # public API: Value, Neuron, Layer, MLP
│   ├── engine.py         # Value — scalar autograd engine
│   └── nn.py             # Neuron, Layer, MLP
├── demo.py               # 20-step training loop (run with: python demo.py)
├── README.md
├── requirements.txt
└── notebooks/
    └── micrograd_from_scratch.ipynb
```

### Project Highlights

- Built a minimal scalar-valued autograd engine from scratch.
- Implemented operator overloading for mathematical expressions such as addition, multiplication, powers, and division.
- Constructed a dynamic computational graph during the forward pass.
- Implemented reverse-mode automatic differentiation using topological sorting.
- Built neural network components including `Neuron`, `Layer`, and `MLP`.
- Trained a small neural network using gradient descent.

### Architecture Overview

The implementation is organized around four levels of abstraction:

```text
Value → Neuron → Layer → MLP
```

Each component has a specific role:

- `Value` is the scalar object that stores data, gradients, operation history, and backward logic.
- `Neuron` is a single neural network unit with weights, bias, and non-linear activation.
- `Layer` is a collection of neurons operating on the same input vector.
- `MLP` is a sequence of layers forming a feed-forward neural network.

### 1. Value: The Atomic Autograd Block

The `Value` class is the core primitive of the entire system.

Each `Value` object stores:

- `data`: the scalar numerical value
- `grad`: the gradient of the final loss with respect to this value
- `_prev`: references to previous `Value` objects used to create it
- `_op`: the operation that produced the current value
- `_backward`: a function that propagates gradients to its parent nodes

Conceptually, every mathematical operation creates a new `Value` node and links it back to the values that produced it. This forms a dynamic computational graph.

Example:

```python
a = Value(2.0)
b = Value(3.0)
c = a * b
```

Here, `c` stores:

```text
data = 6.0
_prev = {a, b}
_op = "*"
```

During backpropagation, `c` knows how to distribute its gradient back to `a` and `b`.

### 2. Neuron: A Single Neural Network Unit

A `Neuron` represents one computational unit in a neural network.

It contains:

- a list of learnable weights
- one learnable bias
- a non-linear activation function

The forward computation is:

$$
y = \tanh(w_1x_1 + w_2x_2 + \cdots + w_nx_n + b)
$$

A simplified implementation:

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
```

Key ideas:

- `nin` defines the number of input features.
- `self.w` contains one weight per input feature.
- `self.b` shifts the activation value.
- `tanh()` introduces non-linearity, allowing the model to learn non-linear patterns.

### 3. Layer: A Group of Neurons

A `Layer` contains multiple neurons that receive the same input vector.

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
```

If an input vector has dimension 2 and passes through `Layer(2, 3)`, the layer produces 3 output values:

```text
Input:  [x1, x2]
Output: [n1(x), n2(x), n3(x)]
```

So the layer transforms:

$$
\mathbb{R}^2 \rightarrow \mathbb{R}^3
$$

This means the layer maps a lower-dimensional input representation into a higher-dimensional learned feature space.

### 4. MLP: Multi-Layer Perceptron

The `MLP` class connects multiple layers sequentially.

```python
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

For example:

```python
n = MLP(3, [4, 4, 1])
```

This creates the following architecture:

```text
Input dimension: 3

Layer 1: 3 → 4
Layer 2: 4 → 4
Layer 3: 4 → 1
```

The list construction:

```python
sz = [nin] + nouts
```

turns:

```python
nin = 3
nouts = [4, 4, 1]
```

into:

```python
sz = [3, 4, 4, 1]
```

Then each adjacent pair defines one layer:

```text
sz[0], sz[1] → Layer(3, 4)
sz[1], sz[2] → Layer(4, 4)
sz[2], sz[3] → Layer(4, 1)
```

The forward pass repeatedly reassigns the output of each layer back to `x`:

```python
x = layer(x)
```

This allows the output of one layer to become the input of the next layer.

### 5. Python Magic Methods

The `Value` class overloads Python operators so that custom scalar objects can behave like normal numbers while still tracking gradients.

For example:

```python
a = Value(2.0)
b = Value(3.0)

c = a + b
d = a * b
e = a ** 2
```

Each operation internally creates a new `Value` object and registers the correct backward function.

#### Addition and Reverse Addition

Python's built-in `sum()` function starts with the integer `0` by default.

So this expression:

```python
sum([Value(1.0), Value(2.0)])
```

starts as:

```python
0 + Value(1.0)
```

Since Python's integer type does not know how to add itself to a custom `Value` object, we implement `__radd__`:

```python
def __radd__(self, other):
    return self + other
```

This allows Python to fall back from:

```python
0 + Value
```

to:

```python
Value + 0
```

This is especially useful when computing neuron activations with `sum()`.

#### Multiplication and Reverse Multiplication

Multiplication also needs both normal and reverse versions:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out

def __rmul__(self, other):
    return self * other
```

This allows both forms to work correctly:

```python
Value(2.0) * 3
3 * Value(2.0)
```

#### Division Through Powers

Division can be implemented using multiplication and negative powers:

$$
\frac{a}{b} = a \cdot b^{-1}
$$

So instead of writing a separate backward rule for division, we can define:

```python
def __truediv__(self, other):
    return self * other ** -1
```

This relies on a correct implementation of `__pow__`:

```python
def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data ** other, (self,), f"**{other}")

    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad

    out._backward = _backward
    return out
```

The derivative follows the power rule:

$$
\frac{d}{dx}x^n = nx^{n-1}
$$

### 6. Training Loop

A typical training loop contains four steps:

1. Forward pass
2. Gradient reset
3. Backward pass
4. Parameter update

```python
for k in range(20):
    # 1. Forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)])

    # 2. Reset gradients
    for p in n.parameters():
        p.grad = 0.0

    # 3. Backward pass
    loss.backward()

    # 4. Gradient descent update
    for p in n.parameters():
        p.data -= 0.05 * p.grad

    print(k, loss.data)
```

### 7. Forward Pass

During the forward pass, the network computes predictions and builds the computational graph.

```python
ypred = [n(x) for x in xs]
loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)])
```

This step does two things at the same time:

1. Computes numerical values through `.data`
2. Builds graph connections through `_prev` and `_backward`

For each training example:

```text
input x → MLP → prediction ypred
```

Then the prediction is compared against the ground-truth label:

```text
prediction ypred → loss
```

The loss is a single scalar value, which is important because backpropagation starts from one final objective.

### 8. Why Backpropagation Starts from Loss

The model does not directly optimize raw predictions. It optimizes the loss function:

$$
L = \sum_i (\hat{y}_i - y_i)^2
$$

The goal is to compute:

$$
\frac{\partial L}{\partial p}
$$

for every parameter `p`.

This tells us how changing each weight or bias affects the final loss.

`ypred` only contains predictions. It does not directly measure how wrong the model is. The `loss` value combines predictions and labels into a single scalar objective, so it provides the correct starting point for gradient computation.

### 9. Backward Pass

Calling:

```python
loss.backward()
```

starts reverse-mode automatic differentiation.

The backward pass works in two stages:

1. Topologically sort all nodes in the computational graph.
2. Visit nodes in reverse order and apply the chain rule.

A simplified version:

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(self)

    self.grad = 1.0

    for node in reversed(topo):
        node._backward()
```

The line:

```python
self.grad = 1.0
```

means:

$$
\frac{\partial L}{\partial L} = 1
$$

This initializes the gradient of the final loss with respect to itself.

### 10. Why Gradients Must Be Reset

Gradients are accumulated using `+=`.

For example:

```python
self.grad += ...
```

This is necessary because one variable may affect the final loss through multiple paths in the computational graph.

However, it also means that gradients from previous training iterations will remain unless we manually clear them.

Therefore, before every backward pass, we reset all gradients:

```python
for p in n.parameters():
    p.grad = 0.0
```

Without this step, gradients from different training iterations would accumulate incorrectly, causing unstable or incorrect updates.

### 11. Parameter Update with Gradient Descent

After gradients are computed, each parameter is updated using gradient descent:

```python
p.data -= learning_rate * p.grad
```

Mathematically:

$$
p_{\text{new}} = p - \eta \frac{\partial L}{\partial p}
$$

where:

- $p$ is a parameter
- $\eta$ is the learning rate
- $\frac{\partial L}{\partial p}$ is the gradient of the loss with respect to the parameter

The gradient points in the direction of steepest increase of the loss. Since we want to minimize the loss, we move in the opposite direction.

### 12. Understanding the Negative Sign in Gradient Descent

Gradient descent uses:

```python
p.data -= learning_rate * p.grad
```

The negative sign is important because the gradient points uphill.

If:

$$
\frac{\partial L}{\partial p} > 0
$$

then increasing `p` increases the loss. To reduce the loss, we should decrease `p`.

If:

$$
\frac{\partial L}{\partial p} < 0
$$

then increasing `p` decreases the loss. To reduce the loss, we should increase `p`.

So the update rule automatically chooses the correct direction.

#### Example: Updating a Negative Parameter

Suppose:

```text
p.data = -0.64922
p.grad = 0.02875
learning_rate = 0.05
```

Then:

$$
p_{\text{new}} = -0.64922 - 0.05 \times 0.02875
$$

$$
p_{\text{new}} = -0.65065
$$

The parameter becomes more negative:

```text
Before: -0.64922
After:  -0.65065
```

On the number line:

```text
Negative direction ←─────────────── 0 ───────────────→ Positive direction

        -0.65065      -0.64922        0
        updated       original
```

This is correct because the positive gradient means moving to the right would increase the loss. Therefore, gradient descent moves the parameter to the left.

### 13. Key Takeaways

This project demonstrates several core deep learning concepts from first principles:

- Neural networks can be built from scalar operations.
- Each scalar operation creates a node in a computational graph.
- Backpropagation is reverse-mode automatic differentiation over this graph.
- Gradients must accumulate because one variable can influence the loss through multiple paths.
- Gradients must also be reset before each new training step.
- Gradient descent updates parameters in the opposite direction of the gradient.
- A small MLP can be implemented using only Python classes, operator overloading, and calculus.

### Reference

This project is inspired by Andrej Karpathy's **micrograd**, a minimal scalar-valued autograd engine for educational purposes.