import math

import torch
from torch import nn
from torch import Tensor


class Linear(nn.Module):
    """
    不带 bias 的线性层。
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                d_out,
                d_in,
                device=device,
                dtype=dtype,
            )
        )

        std = math.sqrt(2.0 / (d_in + d_out))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """
    Token Embedding。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                vocab_size,
                d_model,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(
        self,
        token_ids: Tensor,
    ) -> Tensor:
        return self.weight[token_ids]


def silu(
    x: Tensor,
) -> Tensor:
    """
    SiLU(x) = x * sigmoid(x)
    """

    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.w1 = Linear(
            d_in=d_model,
            d_out=d_ff,
            device=device,
            dtype=dtype,
        )

        self.w2 = Linear(
            d_in=d_ff,
            d_out=d_model,
            device=device,
            dtype=dtype,
        )

        self.w3 = Linear(
            d_in=d_model,
            d_out=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        gate = silu(self.w1(x))
        value = self.w3(x)

        hidden = gate * value

        return self.w2(hidden)


class RMSNorm(nn.Module):
    """
    RMSNorm:

        x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        original_dtype = x.dtype

        x_float = x.to(torch.float32)

        mean_square = x_float.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        rms = torch.sqrt(
            mean_square + self.eps
        )

        normalized = x_float / rms

        output = (
            normalized
            * self.weight.to(torch.float32)
        )

        return output.to(original_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)。
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(
                "d_k must be even."
            )

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        dimension_indices = torch.arange(
            0,
            d_k,
            2,
            dtype=torch.float32,
            device=device,
        )

        inverse_frequencies = (
            theta
            ** (
                -dimension_indices / d_k
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
            device=device,
        )

        angles = torch.outer(
            positions,
            inverse_frequencies,
        )

        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)

        self.register_buffer(
            "cos_cached",
            cos_values,
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            sin_values,
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor,
    ) -> Tensor:
        cos = self.cos_cached[
            token_positions
        ]

        sin = self.sin_cached[
            token_positions
        ]

        cos = cos.to(
            device=x.device,
            dtype=x.dtype,
        )

        sin = sin.to(
            device=x.device,
            dtype=x.dtype,
        )

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        output = torch.stack(
            (
                rotated_even,
                rotated_odd,
            ),
            dim=-1,
        )

        return output.flatten(
            start_dim=-2,
        )


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    softmax(Q K^T / sqrt(d_k)) V
    """

    d_k = Q.shape[-1]

    attention_scores = (
        Q @ K.transpose(-2, -1)
    )

    attention_scores = (
        attention_scores
        / math.sqrt(d_k)
    )

    if mask is not None:
        attention_scores = (
            attention_scores.masked_fill(
                ~mask,
                float("-inf"),
            )
        )

    attention_weights = torch.softmax(
        attention_scores,
        dim=-1,
    )

    return attention_weights @ V


class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = (
            d_model // num_heads
        )

        self.q_proj = Linear(
            d_in=d_model,
            d_out=d_model,
            device=device,
            dtype=dtype,
        )

        self.k_proj = Linear(
            d_in=d_model,
            d_out=d_model,
            device=device,
            dtype=dtype,
        )

        self.v_proj = Linear(
            d_in=d_model,
            d_out=d_model,
            device=device,
            dtype=dtype,
        )

        self.output_proj = Linear(
            d_in=d_model,
            d_out=d_model,
            device=device,
            dtype=dtype,
        )

        if (
            theta is not None
            and max_seq_len is not None
        ):
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        new_shape = (
            *x.shape[:-2],
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        Q = Q.reshape(new_shape)
        K = K.reshape(new_shape)
        V = V.reshape(new_shape)

        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len,
                    device=x.device,
                    dtype=torch.long,
                )

            rope_positions = (
                token_positions.unsqueeze(-2)
            )

            Q = self.rope(
                Q,
                rope_positions,
            )

            K = self.rope(
                K,
                rope_positions,
            )

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )

        attention_output = (
            scaled_dot_product_attention(
                Q=Q,
                K=K,
                V=V,
                mask=causal_mask,
            )
        )

        attention_output = (
            attention_output.transpose(
                -3,
                -2,
            )
        )

        attention_output = (
            attention_output.reshape(
                *x.shape[:-2],
                seq_len,
                self.d_model,
            )
        )

        return self.output_proj(
            attention_output
        )


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Block。

        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.ln1 = RMSNorm(
            d_model=d_model,
            eps=1e-5,
            device=device,
            dtype=dtype,
        )

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(
            d_model=d_model,
            eps=1e-5,
            device=device,
            dtype=dtype,
        )

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        attention_input = self.ln1(x)

        attention_output = self.attn(
            attention_input,
            token_positions=token_positions,
        )

        x = x + attention_output

        ffn_input = self.ln2(x)

        ffn_output = self.ffn(
            ffn_input
        )

        x = x + ffn_output

        return x


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer Language Model。

    流程：

        token ids
            ↓
        Token Embedding
            ↓
        Transformer Block × num_layers
            ↓
        Final RMSNorm
            ↓
        LM Head
            ↓
        logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        # 名字必须是 token_embeddings，
        # 与官方 state_dict 对应
        self.token_embeddings = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        # 多层 TransformerBlock
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # 最后再做一次 RMSNorm
        self.ln_final = RMSNorm(
            d_model=d_model,
            eps=1e-5,
            device=device,
            dtype=dtype,
        )

        # 投影到 vocab_size
        self.lm_head = Linear(
            d_in=d_model,
            d_out=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_indices: Tensor,
    ) -> Tensor:
        """
        in_indices:
            (batch_size, sequence_length)

        return:
            (batch_size, sequence_length, vocab_size)
        """

        seq_len = in_indices.shape[-1]

        if seq_len > self.context_length:
            raise ValueError(
                f"Input sequence length {seq_len} "
                f"exceeds context length "
                f"{self.context_length}."
            )

        # ==========================================
        # 1. Token Embedding
        # ==========================================

        x = self.token_embeddings(
            in_indices
        )

        # ==========================================
        # 2. 当前 token 的 position
        #
        # 0, 1, 2, ..., seq_len - 1
        # ==========================================

        token_positions = torch.arange(
            seq_len,
            device=in_indices.device,
            dtype=torch.long,
        )

        # ==========================================
        # 3. Transformer blocks
        # ==========================================

        for layer in self.layers:
            x = layer(
                x,
                token_positions=token_positions,
            )

        # ==========================================
        # 4. Final RMSNorm
        # ==========================================

        x = self.ln_final(x)

        # ==========================================
        # 5. LM Head
        #
        # d_model -> vocab_size
        # ==========================================

        logits = self.lm_head(x)

        return logits