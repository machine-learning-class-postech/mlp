from __future__ import annotations

from typing import (
    TypedDict,
    Unpack,
    cast,  # type: ignore
)

import numpy as np
from mlp.loss.protocols import Loss
from mlp.module.protocols import Module
from mlp.optimizer.protocols import Optimizer
from mlp.tree.functions import tree_map
from mlp.tree.types import Tree

# Built-in modules


class _Singleton(TypedDict):
    x: np.ndarray


class _Attention(TypedDict):
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray


class Linear(Module[np.ndarray]):
    """
    Linear transformation layer implementing y = x @ W^T + b.

    This is a fully connected layer that performs an affine transformation
    on the input. Supports both forward, backward, and second-order backward passes.

    Args:
        weights: Weight matrix of shape (output_size, input_size)
        bias: Bias vector of shape (output_size,)
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """Return parameters (weights and bias)."""
        return {"weights": self.weights, "bias": self.bias}

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Forward pass: compute y = x @ W^T + b.

        Args:
            x: Input array of shape (B, input_size)

        Returns:
            Returns: Output array of shape (B, output_size)
        """
        x = inputs["x"]
        return x @ self.weights.T + self.bias

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Singleton]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass: compute gradients with respect to parameters and inputs.

        Args:
            gradient_outputs: Gradient of loss with respect to output
            x: Input from forward pass

        Returns:
            Tuple of (parameter_gradients, input_gradients)
        """
        x = inputs["x"]

        weight_grads = gradient_outputs.T @ x
        bias_grads = np.sum(gradient_outputs, axis=0)

        param_grads: Tree[np.ndarray] = {
            "weights": weight_grads,
            "bias": bias_grads,
        }

        input_grads = {"x": gradient_outputs @ self.weights}
        return param_grads, input_grads

    def with_parameters(
        self,
        parameters: Tree[np.ndarray],
    ) -> Linear:
        """Create a new Linear layer with different parameters."""
        assert isinstance(parameters, dict), "Parameters must be a dictionary"
        assert "weights" in parameters and "bias" in parameters, (
            "Parameters must include 'weights' and 'bias'"
        )
        return Linear(**parameters)  # pyright: ignore[reportArgumentType]

    @staticmethod
    def create(
        input_size: int, output_size: int, random_generator: np.random.Generator
    ) -> Linear:
        """
        Factory method to create a Linear layer with random initialization.

        Args:
            input_size: Number of input features
            output_size: Number of output features
            random_generator: NumPy random generator for weight initialization

        Returns:
            New Linear layer with Glorot/Xavier Uniform initialized weights and zero bias
        """
        limit = np.sqrt(6 / (input_size + output_size))

        weights = random_generator.uniform(
            low=-limit, high=limit, size=(output_size, input_size)
        )
        bias = np.zeros(output_size)
        return Linear(weights, bias)


class ReLU(Module[np.ndarray]):
    """
    Rectified Linear Unit (ReLU) activation function.

    Implements f(x) = max(0, x), which sets negative values to zero
    and preserves positive values. This is the most common activation
    function in deep learning due to its simplicity and effectiveness.
    """

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """ReLU has no learnable parameters."""
        return {}

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Forward pass: apply ReLU activation f(x) = max(0, x).

        Args:
            x: Input array

        Returns:
            Output array with negative values clipped to zero
        """
        x = inputs["x"]
        return np.maximum(0, x)

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Singleton]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass: compute ReLU derivative.
        At x = 0, we conventionally use 0 (though mathematically undefined).

        Args:
            gradient_outputs: Gradient of loss with respect to output
            x: Input from forward pass

        Returns:
            Tuple of (empty parameter_gradients, input_gradients)
        """
        x = inputs["x"]
        param_grads: Tree[np.ndarray] = {}
        input_grads = {"x": gradient_outputs * (x > 0)}
        return param_grads, input_grads

    def with_parameters(
        self,
        parameters: Tree[np.ndarray],
    ) -> ReLU:
        """Create new ReLU (parameters ignored since ReLU is parameter-free)."""
        return ReLU()

    @staticmethod
    def create() -> ReLU:
        """Factory method to create a ReLU activation function."""
        return ReLU()


class Softmax(Module[np.ndarray]):
    """
    Softmax activation function.

    Implements the softmax transformation over the last dimension:
        f(x_i) = exp(x_i) / sum_j exp(x_j)

    This operation converts raw logits (unnormalized scores) into a
    probability distribution where all elements are non-negative and
    sum to 1. Numerical stability is ensured by subtracting the maximum value
    in each row (or along the specified axis) before exponentiation.
    """

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """Softmax has no learnable parameters."""
        return {}

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Forward pass: apply softmax transformation.

        Args:
            x: Input array of shape (..., num_classes), typically logits.

        Returns:
            Array of the same shape containing normalized probabilities,
            where each row sums to 1.
        """
        input = inputs["x"]
        exp_inputs = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return exp_inputs / np.sum(exp_inputs, axis=-1, keepdims=True)

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Singleton]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass: compute the gradient of softmax.

        For each input x_i, the Jacobian of the softmax is:
            ∂y_i/∂x_j = y_i (δ_ij - y_j)

        This implementation efficiently computes:
            grad_inputs = y * (gradient_outputs - sum(gradient_outputs * y))

        Args:
            gradient_outputs: Gradient of loss with respect to output.
            x: Input array from forward pass.

        Returns:
            Tuple of (empty parameter_gradients, input_gradients),
            where input_gradients["x"] has the same shape as x.
        """
        x = inputs["x"]
        y = self.forward(x=x)

        gy = np.sum(gradient_outputs * y, axis=-1, keepdims=True)
        grad_inputs = y * (gradient_outputs - gy)

        param_grads: Tree[np.ndarray] = {}
        input_grads = {"x": grad_inputs}
        return param_grads, input_grads

    def with_parameters(
        self,
        parameters: Tree[np.ndarray],
    ) -> Softmax:
        """Create a new Softmax module (no parameters to copy)."""
        return Softmax()

    @staticmethod
    def create() -> Softmax:
        """Factory method to create a Softmax activation function."""
        return Softmax()


class Embedding(Module[np.ndarray]):
    """
    Minimal embedding layer for decoder-only models.

    Stores a weight matrix W ∈ R^{V×D} and performs integer lookups.
    Supports ONLY 1D token indices (T,) and returns embeddings (T, D).
    No padding handling, no batched (B, T) path.

    Args:
        weight (np.ndarray): Embedding weight matrix of shape (V, D).

    Returns:
        np.ndarray: Embedding vectors for the given 1D indices, shape (T, D).
    """

    def __init__(self, weight: np.ndarray):
        """Initialize with a (V, D) weight matrix."""
        if weight.ndim != 2:
            raise ValueError(
                "weight must be 2D of shape (num_embeddings, embedding_dim)"
            )
        self.weight = weight

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """Return the parameter dictionary: {'weight': (V, D)}."""
        return {"weight": self.weight}

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Forward pass: integer lookup for 1D indices.

        Args:
            x (np.ndarray): Integer token indices of shape (T,).

        Returns:
            np.ndarray: Embedding vectors of shape (T, D).

        Raises:
            TypeError: If indices are not integer dtype.
            ValueError: If input is not 1D.
        """
        x = inputs["x"]
        return self.weight[x]

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Singleton]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass: scatter-add into the weight matrix.

        Args:
            gradient_outputs (np.ndarray): dL/dE of shape (T, D), matching forward output.
            x (np.ndarray): Integer indices of shape (T,).

        Returns:
            Tuple:
                - grads (dict): {"weight": dW} with shape (V, D).
                - grad_inputs (dict): {"x": zeros_like(x)} (no gradient to indices).

        Notes:
            - Uses `np.add.at` to correctly accumulate when indices repeat.
        """
        x = inputs["x"]

        grad_weight = np.zeros_like(self.weight)
        np.add.at(grad_weight, x, gradient_outputs)

        grads: Tree[np.ndarray] = {"weight": grad_weight}
        grad_inputs = {"x": np.zeros_like(x)}
        return grads, grad_inputs

    def with_parameters(self, parameters: Tree[np.ndarray]) -> Embedding:
        """
        Return a new Embedding with the provided weight matrix.

        Args:
            parameters (dict): Must contain "weight" with shape (V, D).
        """
        if not isinstance(parameters, dict) or "weight" not in parameters:
            raise ValueError("Parameters must be a dict containing 'weight'.")
        W = np.asarray(parameters["weight"])
        if W.ndim != 2:
            raise ValueError("weight must be 2D.")
        return Embedding(weight=W)

    @staticmethod
    def create(
        input_size: int,
        output_size: int,
        random_generator: np.random.Generator,
    ) -> Embedding:
        """
        Factory: initialize weights with U(-1/sqrt(D), 1/sqrt(D)).

        Args:
            input_size (int): Vocabulary size V.
            output_size (int): Embedding dimension D.
            random_generator (np.random.Generator): RNG for initialization.

        Returns:
            Embedding: New instance with (V, D) weights.
        """
        limit = np.sqrt(3.0 / float(output_size))
        weight = random_generator.uniform(
            low=-limit, high=limit, size=(input_size, output_size)
        )
        return Embedding(weight=weight)


# Transformer modules
# You need to implement these modules.


class LayerNorm(Module[np.ndarray]):
    """
    Layer normalization over the last feature dimension with elementwise-affine transform.

    Normalizes each sample along the last axis to zero mean and unit variance, then applies
    learnable scale (weights) and shift (beta).
        y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

    Args:
        weights(gamma):  Scale vector of shape (D,) matching the last dimension.
        bias(beta):  Shift vector of shape (D,) matching the last dimension.
        eps:  Small constant for numerical stability added to the variance.
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray, eps: float = 1e-5):
        """Initialize LayerNorm with weights(gamma), bias(beta), and epsilon (numerical stability)."""
        ...

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """Return learnable parameters {"weights": (D,), "bias": (D,)}."""
        ...

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Forward pass: Stats are computed over axis=-1; weights(γ) and bias(β) are broadcast over leading axes
            y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

        Args:
            x: Input array of shape (..., D).

        Returns:
            Output array of shape (..., D) with per-sample normalized features.
        """
        ...

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Singleton]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Compute gradients w.r.t. weights, bias, and input without relying on forward-time caches.

        Args:
            gradient_outputs: dL/dy of shape (..., D).
            x: Forward input of shape (..., D).

        Returns:
            Tuple of:
                - parameter_gradients: {"weights": (D,), "bias": (D,)}
                - input_gradients: {"x": dL/dx of shape (..., D)}
        """
        ...

    def with_parameters(
        self,
        parameters: Tree[np.ndarray],
    ) -> LayerNorm: ...

    @staticmethod
    def create(output_size: int) -> LayerNorm:
        """
        Factory: create LayerNorm with weights=ones(D) and bias=zeros(D).

        Args:
            output_size: Feature size D of the last dimension.

        Returns:
            New LayerNorm instance with default-initialized parameters.
        """
        weights = np.ones(output_size)
        bias = np.zeros(output_size)
        return LayerNorm(weights, bias)


class FeedForward(Module[np.ndarray]):
    """
    Two-layer feed-forward neural network (FFN) module.

    Implements the standard position-wise feed-forward block used in Transformers:
        FFN(x) = W₂ * ReLU(W₁x + b₁) + b₂

    Args:
        linear1: First Linear layer (input → hidden)
        activation: ReLU function
        linear2: Second Linear layer (hidden → output)
    """

    def __init__(self, layer1: Linear, layer2: Linear): ...

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """
        Return parameters from both linear layers.
        Format:
            {"layer1": {...}, "layer2": {...}}
        """
        ...

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Compute forward pass of the FFNN.

            y = W₂ * ReLU(W₁x + b₁) + b₂

        Args:
            x: Input array of shape (..., input_dim)

        Returns:
            Output array of shape (..., output_dim)
        """
        ...

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Singleton]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass without relying on forward-time caches.
        Recomputes needed intermediates from x, then propagates:
            layer2 ← ReLU ← layer1

        Returns:
            Tuple containing:
                - parameter_gradients: a dict with layer1 and layer2
                - input_gradients: a dict with a gradient for {"x"}
        """
        ...

    def with_parameters(
        self,
        parameters: Tree[np.ndarray],
    ) -> FeedForward:
        """Return a new FFN with updated layer parameters."""
        ...

    @staticmethod
    def create(
        input_size: int,
        hidden_size: int,
        output_size: int,
        random_generator: np.random.Generator,
    ) -> FeedForward:
        """
        Factory method to create a two-layer FFN with ReLU activation.

        Args:
            input_size: Dimensionality of input features.
            hidden_size: Number of hidden units in the intermediate layer.
            output_size: Dimensionality of output features.
            random_generator: NumPy random generator for weight initialization.

        Returns:
            New FeedForwardNetwork instance with Linear–ReLU–Linear structure.
        """
        layer1 = Linear.create(
            input_size=input_size,
            output_size=hidden_size,
            random_generator=random_generator,
        )
        layer2 = Linear.create(
            input_size=hidden_size,
            output_size=output_size,
            random_generator=random_generator,
        )
        return FeedForward(layer1, layer2)


class PositionalEncoding(Module[np.ndarray]):
    """
    Sinusoidal positional encoding.

    Generates a fixed (non-learnable) matrix `PE ∈ ℝ^{L×D}` with:
        PE[n, 2k]   = sin(n / base^{2k/D})
        PE[n, 2k+1] = cos(n / base^{2k/D})

    Args:
        seq_len (int): Maximum sequence length L.
        d_model (int): Model dimension D (must match input's last dim).
        base (float): Geometric base for wavelengths (default: 10000.0).
    """

    def __init__(self, seq_len: int, d_model: int, base: float = 10000.0): ...

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """No learnable parameters; returns an empty dict."""
        ...

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Add positional encodings to a (T, D) float tensor.

        Computes:
            y[0:T] = x + PE[0:T]

        Args:
            x: Float array of shape (T, D) with D == d_model and T ≤ seq_len.

        Returns:
            Array of shape (T, D) with positions added elementwise.

        Raises:
            ValueError: If dtype/shape constraints are violated.
        """
        ...

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Singleton]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Pass-through gradient for addition with no parameters.

        Args:
            gradient_outputs: dL/dy of shape (T, D).
            x: Forward input of shape (T, D).

        Returns:
            Tuple of:
                - parameter_gradients: {} (no parameters)
                - input_gradients: {"x": dL/dx} where dL/dx = gradient_outputs
        """
        ...

    def with_parameters(self, parameters: Tree[np.ndarray]) -> PositionalEncoding:
        """Return a new PositionalEncoding with the same hyperparameters (no parameters to copy)."""
        ...

    @staticmethod
    def create(
        input_size: int, output_size: int, random_generator: np.random.Generator
    ) -> PositionalEncoding:
        """
        Factory: create a PositionalEncoding.

        Args:
            input_size: Maximum sequence length L.
            output_size: Model dimension D.
            random_generator: Unused; kept for interface compatibility.

        Returns:
            PositionalEncoding with seq_len=L and d_model=D.
        """
        return PositionalEncoding(seq_len=input_size, d_model=output_size, base=10000.0)


class ScaledSelfAttention(Module[np.ndarray]):
    """
    Scaled dot-product self-attention with optional causal masking.

    Computes:
        Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) + M_causal ) V

    Implementation notes:
        - If `causal=True`, applies a strict upper-triangular mask (positions j > i are blocked).
        - Masked logits receive a large negative bias (-1e9) before softmax: a numerically
          stable proxy for -inf that drives the corresponding probabilities to ~0.

    Args:
        W_q (np.ndarray): Query projection matrix of shape (d_in, d_k)
        W_k (np.ndarray): Key   projection matrix of shape (d_in, d_k)
        W_v (np.ndarray): Value projection matrix of shape (d_in, d_v)
        causal (bool): If True, enable causal masking; otherwise no masking.
    """

    def __init__(
        self, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, *, causal: bool = False
    ): ...

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """Return learnable projection matrices for Q, K, and V."""
        ...

    def forward(self, **inputs: Unpack[_Attention]) -> np.ndarray:
        """
        Forward pass; applies causal masking only if `self.causal` is True.

        Args:
            q, k, v (np.ndarray): Input matrices of shape (T, d_in)

        Returns:
            np.ndarray: Output matrix of shape (T, d_v)
        """
        ...

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Attention]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass mirroring the forward masking logic.

        Args:
            gradient_outputs (np.ndarray): dL/doutput
            q, k, v (np.ndarray): Input matrices of shape (T, d_in)

        Returns:
            Tuple containing:
                - parameter_gradients: {"W_q", "W_k", "W_v"}
                - input_gradients: {"q", "k", "v"}
        """
        ...

    def with_parameters(self, parameters: Tree[np.ndarray]) -> ScaledSelfAttention:
        """Return a new ScaledSelfAttention with updated parameter matrices."""
        ...

    @staticmethod
    def create(
        input_size: int,
        output_size: int,
        random_generator: np.random.Generator,
        *,
        causal: bool = True,
    ) -> ScaledSelfAttention:
        """
        Factory: Xavier/Glorot uniform init for Q/K/V. Causal flag controls masking.

        Args:
            input_size (int): d_in
            output_size (int): d_k / d_v per head
            random_generator (np.random.Generator): RNG for initialization
            causal (bool): If True, enable causal masking

        Returns:
            ScaledSelfAttention: New instance.
        """
        limit = np.sqrt(6.0 / (input_size + output_size))
        W_q = random_generator.uniform(
            low=-limit, high=limit, size=(input_size, output_size)
        )
        W_k = random_generator.uniform(
            low=-limit, high=limit, size=(input_size, output_size)
        )
        W_v = random_generator.uniform(
            low=-limit, high=limit, size=(input_size, output_size)
        )
        return ScaledSelfAttention(W_q=W_q, W_k=W_k, W_v=W_v, causal=causal)


class MultiHeadAttention(Module[np.ndarray]):
    """
    Multi-head self-attention.

    Heads honor the `causal` flag set at construction time.
    Forward/backward expect q, k, v only (no mask-related runtime arguments).

    Inputs q, k, v are unprojected sequences of shape (T, d_model).

    Args:
        heads (list[ScaledSelfAttention]): Attention heads
            Each head contains its own linear projections (W_q, W_k, W_v).
        W_out (np.ndarray): Output projection matrix of shape (h * d_v, d_model)

    Outputs are concatenated and projected with W_out.
    """

    def __init__(self, heads: list[ScaledSelfAttention], W_out: np.ndarray): ...

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """
        Return parameters of all attention heads and output projection.
        Format:
            {"head_0": {...}, "head_1": {...}, ..., "W_out": np.ndarray}
        """
        ...

    def forward(self, **inputs: Unpack[_Attention]) -> np.ndarray:
        """
        Forward pass.

        MHA(Q, K, V) = concat(head_1, ..., head_h) * W_O
        Args:
            q, k, v (np.ndarray): Inputs of shape (T, d_model)

        Returns:
            np.ndarray: Output of shape (T, d_model)
        """
        ...

    def backward(
        self, gradient_outputs: np.ndarray, **inputs: Unpack[_Attention]
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass through all heads and output projection.

        Args:
            gradient_outputs (np.ndarray): dL/doutput
            q, k, v (np.ndarray): Inputs of shape (T, d_model)

        Returns:
            Tuple containing:
                - parameter_gradients: {"head_i": {...}, ..., "W_out": ...}
                - input_gradients: {"q", "k", "v"}
        """
        ...

    def with_parameters(self, parameters: Tree[np.ndarray]) -> MultiHeadAttention:
        """Return new MultiHeadAttention with updated parameters for each head and W_out."""
        ...

    @staticmethod
    def create(
        input_size: int,
        embed_dim: int,
        num_heads: int,
        random_generator: np.random.Generator,
        *,
        causal: bool = False,
    ) -> MultiHeadAttention:
        """
        Factory: evenly split heads; heads honor the `causal` flag.
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        d_head = embed_dim // num_heads
        heads = [
            ScaledSelfAttention.create(
                input_size=input_size,
                output_size=d_head,
                random_generator=random_generator,
                causal=causal,
            )
            for _ in range(num_heads)
        ]
        limit = np.sqrt(6.0 / (embed_dim + embed_dim))  # == sqrt(3 / embed_dim)
        W_out = random_generator.uniform(
            low=-limit, high=limit, size=(embed_dim, embed_dim)
        )
        return MultiHeadAttention(heads=heads, W_out=W_out)


class DecoderLayer(Module[np.ndarray]):
    """
    Transformer decoder-only layer: self-attn → Add&Norm → FFN → Add&Norm.

    Uses residual connections and layer normalization around each sublayer. This block
    performs **only self-attention** over the decoder states (no encoder–decoder attention).

    Args:
        mha (MultiHeadAttention): Self-attention over decoder states (T, D).
        norm1 (LayerNorm): Layer normalization after the first residual block.
        ffn  (FeedForward): Position-wise FFN mapping (T, D) → (T, D).
        norm2 (LayerNorm): Layer normalization after the second residual block.
    """

    def __init__(
        self,
        mha: MultiHeadAttention,
        norm1: LayerNorm,
        ffn: FeedForward,
        norm2: LayerNorm,
    ):
        """Initialize the decoder-only submodules (self-attn, norm1, FFN, norm2)."""
        ...

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """
        Return parameter tree for MHA, Norm1, FFN, and Norm2.
        Format:
            {"MHA": {...}, "Norm1": {...}, "FFN": {...}, "Norm2": {...}}
        """
        ...

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Forward pass of the decoder-only layer.

        Args:
            x (np.ndarray): Decoder inputs of shape (T, D).

        Returns:
            np.ndarray: Output of shape (T, D).
        """
        ...

    def backward(
        self,
        gradient_outputs: np.ndarray,
        **inputs: Unpack[_Singleton],
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass of the decoder-only layer.

        Args:
            gradient_outputs (np.ndarray): dL/dy of shape (T, D).
            x (np.ndarray): Forward input (T, D).

        Backward pass of the decoder-only layer.

        Args:
            gradient_outputs (np.ndarray): dL/dy of shape (T, D).
            x (np.ndarray): Forward input (T, D).

        Returns:
            tuple[Tree[np.ndarray], dict[str, np.ndarray]]: (
                parameter_gradients,
                input_gradients {"x": dL/dx (T, D)}
            )
            parameter_gradients = {
                "MHA": {...},
                "Norm1": {...},
                "FFN": {...},
                "Norm2": {...}
            }
            input_gradients = {
                "x": dL/dx (T, D)
            }
        """
        ...

    def with_parameters(self, parameters: Tree[np.ndarray]) -> DecoderLayer:
        """Return a new DecoderLayer with updated parameters for each submodule."""
        ...

    @staticmethod
    def create(
        embed_dim: int,
        num_heads: int,
        hidden_size: int,
        random_generator: np.random.Generator,
        *,
        causal: bool = True,
    ) -> DecoderLayer:
        """
        Factory: decoder-only layer with optional causal self-attention.
        """
        mha = MultiHeadAttention.create(
            input_size=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            random_generator=random_generator,
            causal=causal,
        )
        norm1 = LayerNorm.create(output_size=embed_dim)
        ffn = FeedForward.create(
            input_size=embed_dim,
            hidden_size=hidden_size,
            output_size=embed_dim,
            random_generator=random_generator,
        )
        norm2 = LayerNorm.create(output_size=embed_dim)
        return DecoderLayer(mha, norm1, ffn, norm2)


class Decoder(Module[np.ndarray]):
    """
    Transformer Decoder-only (NumPy).

    Token indices `(T,)` are embedded and position-encoded, then passed through
    a stack of decoder-only layers (self-attn + FFN). The final linear head maps hidden states
    to vocabulary logits `(T, V)`.

    Inputs (**for forward/backward**):
        x   : (T,) int token indices
    Output:
        logits : (T, V)
    """

    def __init__(
        self,
        token_embedding: Embedding,
        pos_encoding: PositionalEncoding,
        layers: list[DecoderLayer],
        lm_head: Linear,
    ):
        """Initialize token embedding, positional encoding, decoder layers, and LM head."""
        ...

    @property
    def parameters(self) -> Tree[np.ndarray]:
        """
        Return nested parameter tree for token embedding, positional encoding, each layer, and the LM head.
        Format:
            {"token": {...},
             "pe": {...},
             "layer_0": {...},
             ...,
             "layer_N": {...},
             "lm_head": {...}}
        """
        ...

    def forward(self, **inputs: Unpack[_Singleton]) -> np.ndarray:
        """
        Forward pass of the decoder-only model.

        Args:
            x (np.ndarray): Integer token indices of shape (T,).

        Returns:
            np.ndarray: Logits of shape (T, V).
        """
        ...

    def backward(
        self,
        gradient_outputs: np.ndarray,
        **inputs: Unpack[_Singleton],
    ) -> tuple[Tree[np.ndarray], dict[str, np.ndarray]]:
        """
        Backward pass of the decoder-only model.

        Unrolls the forward sequence to propagate gradients back through the LM head,
        the stack of decoder-only layers, positional encoding, and the token embedding.

        Args:
            gradient_outputs (np.ndarray): dL/dlogits of shape (T, V).
            x (np.ndarray): Integer token indices of shape (T,).

        Returns:
            tuple[Tree[np.ndarray], dict[str, np.ndarray]]: (
                parameter_gradients,
                input_gradients {"x": zeros_like(x)}
            )
            parameter_gradients = {
                "token": {...},
                "pe": {...},
                "layer_0": {...},
                ...,
                "layer_N": {...},
                "lm_head": {...}
            }
            input_gradients = {
                "x": dL/dx (zeros since x is integer indices)
            }
        """
        ...

    def with_parameters(self, parameters: Tree[np.ndarray]) -> Decoder:
        """Return a new Decoder with updated parameters for token embedding, PE, layers, and LM head."""
        ...

    @staticmethod
    def create(
        input_size: int,
        hidden_size: int,
        output_size: int,
        random_generator: np.random.Generator,
        num_heads: int,
        seq_len: int = 512,
        *,
        causal: bool = True,
    ) -> Decoder:
        """
        Factory: create a minimal 2-layer Decoder (optional causal self-attention).
        """
        vocab_size = input_size
        d_model = output_size

        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        tok_emb = Embedding.create(
            input_size=vocab_size,
            output_size=d_model,
            random_generator=random_generator,
        )
        pos_enc = PositionalEncoding.create(
            input_size=seq_len, output_size=d_model, random_generator=random_generator
        )
        layers = [
            DecoderLayer.create(
                embed_dim=d_model,
                num_heads=num_heads,
                hidden_size=hidden_size,
                random_generator=random_generator,
                causal=causal,
            )
            for _ in range(2)
        ]
        lm_head = Linear.create(
            input_size=d_model,
            output_size=vocab_size,
            random_generator=random_generator,
        )
        return Decoder(tok_emb, pos_enc, layers, lm_head)


# Optimizers and Loss Functions for Evaluation
# Not need to implement below modules.


class Adam(Optimizer[np.ndarray]):
    """
    Adam optimizer (immutable state pattern).

    Keeps first and second moment estimates per-parameter as Trees.
    Returns a NEW Adam instance with updated state at each `step`.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        t: int = 0,
        m: Tree[np.ndarray] | None = None,
        v: Tree[np.ndarray] | None = None,
    ):
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = int(t)
        self.m = m
        self.v = v

    def step(
        self,
        parameters: Tree[np.ndarray],
        gradients: Tree[np.ndarray],
    ) -> tuple[Tree[np.ndarray], Optimizer[np.ndarray]]:
        """
        Perform one Adam update and return (new_parameters, new_optimizer).
        """

        # Lazy init state to zeros with same tree structure
        if self.m is None or self.v is None:
            m0 = tree_map(lambda p: np.zeros_like(p, dtype=p.dtype), parameters)
            v0 = tree_map(lambda p: np.zeros_like(p, dtype=p.dtype), parameters)
        else:
            m0 = self.m
            v0 = self.v

        b1 = self.beta1
        b2 = self.beta2

        # Update biased first/second moments
        m1 = tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, m0, gradients)
        v1 = tree_map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), v0, gradients)

        # Time step
        t1 = self.t + 1
        b1_corr = 1.0 - (b1**t1)
        b2_corr = 1.0 - (b2**t1)

        # Bias-corrected moments
        m_hat = tree_map(lambda m: m / b1_corr, m1)
        v_hat = tree_map(lambda v: v / b2_corr, v1)

        # Parameter update
        def adam_update(p: np.ndarray, mh: np.ndarray, vh: np.ndarray) -> np.ndarray:
            return p - self.learning_rate * mh / (np.sqrt(vh) + self.eps)

        new_parameters = tree_map(adam_update, parameters, m_hat, v_hat)

        # Return new optimizer instance carrying updated state
        new_optimizer = Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            t=t1,
            m=m1,
            v=v1,
        )
        return new_parameters, new_optimizer


class CrossEntropyLoss(Loss[np.ndarray]):
    """
    Multiclass cross-entropy with numerical stability.

    Supports:
      - targets as one-hot: shape (N, C)
      - targets as int class indices: shape (N,)
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Args:
            predictions: logits, shape (N, C)
            targets: one-hot (N, C) or int indices (N,)

        Returns:
            Scalar mean cross-entropy loss.
        """
        logits = predictions  # (N, C)
        # log-softmax (stable)
        m = np.max(logits, axis=1, keepdims=True)  # (N, 1)
        logsumexp = m + np.log(
            np.sum(np.exp(logits - m), axis=1, keepdims=True)
        )  # (N, 1)
        log_probs = logits - logsumexp  # (N, C)

        if targets.ndim == 1:
            # integer labels
            N = logits.shape[0]
            loss = -np.mean(log_probs[np.arange(N), targets])  # type:ignore
            return float(loss)  # pyright: ignore[reportUnknownArgumentType]
        elif targets.ndim == 2:
            # one-hot labels
            per_ex = -np.sum(targets * log_probs, axis=1)  # (N,)
            return float(np.mean(per_ex))  # pyright: ignore[reportUnknownArgumentType]
        else:
            raise ValueError("targets must be (N,) int or (N, C) one-hot.")

    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Gradient w.r.t. logits.

        Args:
            predictions: logits, shape (N, C)
            targets: one-hot (N, C) or int indices (N,)

        Returns:
            dL/dlogits, shape (N, C)
        """
        logits = predictions
        m = np.max(logits, axis=1, keepdims=True)
        expz = np.exp(logits - m)
        softmax = expz / np.sum(expz, axis=1, keepdims=True)  # (N, C)

        N = logits.shape[0]
        if targets.ndim == 1:
            grad = softmax
            grad[np.arange(N), targets] -= 1.0
            grad /= N
            return grad
        elif targets.ndim == 2:
            grad = (softmax - targets) / N
            return grad
        else:
            raise ValueError("targets must be (N,) int or (N, C) one-hot.")
