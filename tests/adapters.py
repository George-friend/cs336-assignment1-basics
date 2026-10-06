from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer,
    compute the transformation of a batched input.
    """

    from cs336_basics.model import Linear

    linear = Linear(
        d_in=d_in,
        d_out=d_out,
        device=weights.device,
        dtype=weights.dtype,
    )

    # 官方测试已经给了固定 weights。
    # 所以覆盖掉我们初始化出来的随机权重。
    with torch.no_grad():
        linear.weight.copy_(weights)

    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer,
    get embeddings for token ids.
    """

    from cs336_basics.model import Embedding

    embedding = Embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        device=weights.device,
        dtype=weights.dtype,
    )

    with torch.no_grad():
        embedding.weight.copy_(weights)

    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of a SwiGLU network,
    return the output using those weights.
    """

    from cs336_basics.model import SwiGLU

    swiglu = SwiGLU(
        d_model=d_model,
        d_ff=d_ff,
        device=in_features.device,
        dtype=in_features.dtype,
    )

    with torch.no_grad():
        swiglu.w1.weight.copy_(w1_weight)
        swiglu.w2.weight.copy_(w2_weight)
        swiglu.w3.weight.copy_(w3_weight)

    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors,
    return scaled dot-product attention.
    """

    from cs336_basics.model import (
        scaled_dot_product_attention,
    )

    return scaled_dot_product_attention(
        Q=Q,
        K=K,
        V=V,
        mask=mask,
    )


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[
        Tensor,
        " ... sequence_length d_model"
    ],
) -> Float[
    Tensor,
    " ... sequence_length d_model"
]:
    """
    Run causal multi-head self-attention without RoPE.
    """

    from cs336_basics.model import (
        MultiHeadSelfAttention,
    )

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        device=in_features.device,
        dtype=in_features.dtype,
    )

    with torch.no_grad():
        attention.q_proj.weight.copy_(
            q_proj_weight
        )

        attention.k_proj.weight.copy_(
            k_proj_weight
        )

        attention.v_proj.weight.copy_(
            v_proj_weight
        )

        attention.output_proj.weight.copy_(
            o_proj_weight
        )

    return attention(
        in_features
    )


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[
        Tensor,
        " ... sequence_length d_model"
    ],
    token_positions: Int[
        Tensor,
        " ... sequence_length"
    ]
    | None = None,
) -> Float[
    Tensor,
    " ... sequence_length d_model"
]:
    """
    Run causal multi-head self-attention with RoPE.
    """

    from cs336_basics.model import (
        MultiHeadSelfAttention,
    )

    attention = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
        device=in_features.device,
        dtype=in_features.dtype,
    )

    with torch.no_grad():
        attention.q_proj.weight.copy_(
            q_proj_weight
        )

        attention.k_proj.weight.copy_(
            k_proj_weight
        )

        attention.v_proj.weight.copy_(
            v_proj_weight
        )

        attention.output_proj.weight.copy_(
            o_proj_weight
        )

    return attention(
        in_features,
        token_positions=token_positions,
    )


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[
        Tensor,
        " ... sequence_length d_k"
    ],
    token_positions: Int[
        Tensor,
        " ... sequence_length"
    ],
) -> Float[
    Tensor,
    " ... sequence_length d_k"
]:
    """
    Apply Rotary Position Embedding to the input tensor.
    """

    from cs336_basics.model import (
        RotaryPositionalEmbedding,
    )

    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_query_or_key.device,
    )

    return rope(
        in_query_or_key,
        token_positions,
    )


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[
        Tensor,
        " batch sequence_length d_model"
    ],
) -> Float[
    Tensor,
    " batch sequence_length d_model"
]:
    """
    Given the weights of a pre-norm Transformer block
    and input features, return the output of running
    the Transformer block.

    This implementation uses RoPE.
    """

    from cs336_basics.model import TransformerBlock

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        device=in_features.device,
        dtype=in_features.dtype,
    )

    # 我们的模块命名和官方权重 key 一致，
    # 因此可以直接加载整个 state dict。
    block.load_state_dict(weights)

    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[
        Tensor,
        " batch_size sequence_length"
    ],
) -> Float[
    Tensor,
    " batch_size sequence_length vocab_size"
]:
    """
    Given the weights of a Transformer language model
    and input indices, run a forward pass.

    This implementation uses RoPE.
    """

    from cs336_basics.model import TransformerLM

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=in_indices.device,
        dtype=weights[
            "token_embeddings.weight"
        ].dtype,
    )

    model.load_state_dict(weights)

    return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.
    """

    from cs336_basics.model import RMSNorm

    rmsnorm = RMSNorm(
        d_model=d_model,
        eps=eps,
        device=weights.device,
        dtype=weights.dtype,
    )

    with torch.no_grad():
        rmsnorm.weight.copy_(weights)

    return rmsnorm(in_features)

def run_silu(
    in_features: Float[Tensor, " ..."],
) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output
    of applying SiLU to each element.
    """

    from cs336_basics.model import silu

    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset and desired batch size/context length,
    sample language-modeling input sequences and labels.
    """

    from cs336_basics.data import get_batch

    return get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )


def run_softmax(
    in_features: Float[Tensor, " ..."],
    dim: int,
) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output
    of softmaxing the specified dimension.
    """

    from cs336_basics.nn_utils import softmax

    return softmax(
        in_features,
        dim=dim,
    )


def run_cross_entropy(
    inputs: Float[
        Tensor,
        " batch_size vocab_size"
    ],
    targets: Int[
        Tensor,
        " batch_size"
    ],
) -> Float[Tensor, ""]:
    """
    Compute average cross-entropy loss.
    """

    from cs336_basics.nn_utils import (
        cross_entropy,
    )

    return cross_entropy(
        inputs,
        targets,
    )


def run_gradient_clipping(
    parameters: Iterable[
        torch.nn.Parameter
    ],
    max_l2_norm: float,
) -> None:
    """
    Clip the combined gradient L2 norm.
    """

    from cs336_basics.nn_utils import (
        gradient_clipping,
    )

    gradient_clipping(
        parameters,
        max_l2_norm,
    )


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer
    that implements AdamW.
    """

    from cs336_basics.optimizer import AdamW

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Return the learning rate at iteration `it`
    under a linear-warmup + cosine-decay schedule.
    """

    from cs336_basics.optimizer import (
        get_lr_cosine_schedule,
    )

    return get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Serialize model, optimizer, and iteration.
    """

    from cs336_basics.serialization import (
        save_checkpoint,
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        out=out,
    )


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Restore model and optimizer state
    from a checkpoint and return iteration.
    """

    from cs336_basics.serialization import (
        load_checkpoint,
    )

    return load_checkpoint(
        src=src,
        model=model,
        optimizer=optimizer,
    )


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    from cs336_basics.tokenizer import Tokenizer

    return Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
    )


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    from cs336_basics.train_bpe import train_bpe

    return train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )