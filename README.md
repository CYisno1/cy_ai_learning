# CY AI Learning

A collection of machine learning, deep learning, and AI engineering projects built while studying core AI concepts from first principles.

## Projects

### 🧠 Neural Networks / Micrograd

A from-scratch implementation of a tiny scalar-valued automatic differentiation engine and neural network library, inspired by Andrej Karpathy's **micrograd**.

This project demonstrates:

- scalar-based automatic differentiation
- dynamic computational graphs
- reverse-mode backpropagation
- gradient descent optimization
- modular neural network components: `Value`, `Neuron`, `Layer`, and `MLP`
- clean Python package structure separated from the original notebook

**Project folder:** [`Neural networks`](./Neural%20networks)  
**Detailed README:** [`Neural networks/README.md`](./Neural%20networks/README.md)

```text
Neural networks/
├── micrograd/
│   ├── __init__.py
│   ├── engine.py
│   └── nn.py
├── demo.py
├── image.png
├── notebooks/
│   └── micrograd_from_scratch.ipynb
├── README.md
└── requirements.txt
```

## Repository Structure

```text
cy_ai_learning/
├── Neural networks/
│   └── Micrograd project
├── README.md
└── .gitignore
```

More projects will be added as this repository grows.