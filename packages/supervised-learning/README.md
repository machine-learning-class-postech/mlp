# Assignment: Supervised Learning

This assignment covers fundamental components of supervised learning for a machine learning course at POSTECH.

## Overview

The assignment covers core concepts in supervised learning including:

- **Neural network layers**: Linear transformations and activation functions
- **Optimization algorithms**: Stochastic Gradient Descent and Newton's method
- **Loss functions**: Regression and classification objectives

## Tasks Breakdown

### Linear Layer (`Linear`)

A fully connected layer that performs affine transformations.

**Features:**

- Forward and backward passes
- Second-order computations for Newton optimization
- Random weight initialization

### ReLU Activation (`ReLU`)

Rectified Linear Unit activation function.

**Features:**

- Non-linear activation function
- Parameter-free implementation

### Stochastic Gradient Descent (`StochasticGradientDescent`)

Standard first-order optimization algorithm.

**Features:**

- Configurable learning rate
- Tree-based parameter updates
- Immutable optimizer instances

### Newton Optimizer (`Newton`)

Second-order optimization using Hessian information.

**Features:**

- Uses Hessian approximation
- Numerical stability considerations

### Mean Squared Error (`MeanSquaredError`)

Regression loss function.

**Features:**

- Loss function for continuous targets

### Cross Entropy Loss (`CrossEntropy`)

Classification loss with softmax.

**Features:**

- Numerical stability considerations
- Combined softmax and cross-entropy computation

## Technical Details

### Tree Operations

The implementation uses tree utility functions from `mlp.tree.functions` for parameter management.

### Protocols and Interfaces

The implementation's components must conform to a set of standardized protocols to ensure interoperability and facilitate automated grading.

- **Layers**: Implement the `Module` and `SecondOrderModule` protocols.
- **Optimizers**: Implement the `Optimizer` and `SecondOrderOptimizer` protocols.
- **Loss Functions**: Implement the `Loss` protocol.

Adherence to these interfaces is mandatory. Students are required to consult the docstrings in the `mlp` package for detailed specifications. Failure to comply with these protocols will result in incompatibility with the automated grading system.

## Testing Framework

The assignment incorporates a comprehensive testing framework divided into two distinct categories. Any attempt to circumvent or interfere with these testing mechanisms (including but not limited to monkey patching the testing framework) constitutes academic misconduct and will be treated accordingly.

### Public Tests (`test_*_public.py`)

- **Purpose**: Student practice and development
- Basic functionality verification
- Shape and type checking
- Simple computational correctness

### Private Tests (`test_*_private.py`)

- **Purpose**: Grading
- Advanced edge cases and numerical stability
- Mathematical correctness verification
- Integration testing across components
- May include public tests as well

## Assignment Objectives

Students demonstrate understanding of:

1. **Linear algebra**: Matrix operations and transformations
2. **Calculus**: Derivatives and chain rule application  
3. **Optimization**: First and second-order methods
4. **Software engineering**: Clean, tested, documented code
5. **Numerical computing**: Stability and precision considerations

## Submission

1. Fill in the code in [`supervised_learning/__init__.py`](supervised_learning/__init__.py).
2. Validate your implementation by running the tests in the `tests` directory using pytest.
    1. Please ensure all public tests pass before submitting your assignment.
3. Submit only [`supervised_learning/__init__.py`](__init__.py) to the course platform.
    1. Submissions in any other format will not be accepted and will result in a score of zero.

## Credits

- Minjae Gwon
  - <https://bxta.kr>
  - <https://github.com/betarixm>
  - <mailto:minjae.gwon@postech.ac.kr>
