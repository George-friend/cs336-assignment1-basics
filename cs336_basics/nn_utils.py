from collections.abc import Iterable

import torch
from torch import Tensor


def softmax(
    x: Tensor,
    dim: int,
) -> Tensor:
    """
    数值稳定版 Softmax。

    softmax(x_i)
        = exp(x_i) / sum_j exp(x_j)

    为了避免 exp(x) 数值溢出，
    先减去这一维上的最大值。
    """

    # (..., n)
    max_values = torch.max(
        x,
        dim=dim,
        keepdim=True,
    ).values

    shifted = x - max_values

    exp_values = torch.exp(shifted)

    denominator = torch.sum(
        exp_values,
        dim=dim,
        keepdim=True,
    )

    return exp_values / denominator


def cross_entropy(
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    计算一个 batch 的平均 Cross Entropy。

    inputs:
        (batch_size, vocab_size)

        未经过 softmax 的 logits

    targets:
        (batch_size,)

        每个样本对应的正确类别 id

    return:
        scalar
    """

    # ----------------------------------------------
    # Cross Entropy:
    #
    # -log softmax(x)[target]
    #
    # =
    #
    # log(sum(exp(x))) - x_target
    #
    # 为了数值稳定，使用 max trick。
    # ----------------------------------------------

    max_logits = torch.max(
        inputs,
        dim=-1,
        keepdim=True,
    ).values

    shifted_logits = (
        inputs - max_logits
    )

    log_sum_exp = (
        torch.log(
            torch.sum(
                torch.exp(shifted_logits),
                dim=-1,
            )
        )
        + max_logits.squeeze(-1)
    )

    # 取得每个样本正确类别对应的 logit
    target_logits = inputs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = (
        log_sum_exp
        - target_logits
    )

    # 对整个 batch 求平均
    return losses.mean()


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """
    把所有参数梯度的整体 L2 norm
    限制在 max_l2_norm 以内。

    修改 parameter.grad 本身，不返回新梯度。
    """

    if max_l2_norm <= 0:
        raise ValueError(
            "max_l2_norm must be positive."
        )

    # iterable 可能是 generator，
    # 所以先变成 list，避免只能遍历一次。
    parameters = list(parameters)

    # 只考虑真正存在梯度的参数。
    grads = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]

    # 没有梯度就什么也不做。
    if len(grads) == 0:
        return

    # ----------------------------------------------
    # 整体梯度 L2 norm:
    #
    # sqrt(
    #     sum over all parameters
    #     sum over all elements
    #     grad^2
    # )
    # ----------------------------------------------

    total_squared_norm = torch.zeros(
        (),
        device=grads[0].device,
        dtype=torch.float32,
    )

    for grad in grads:
        grad_float = grad.detach().to(
            torch.float32
        )

        total_squared_norm = (
            total_squared_norm
            + torch.sum(
                grad_float * grad_float
            )
        )

    total_norm = torch.sqrt(
        total_squared_norm
    )

    # 和 PyTorch clip_grad_norm_ 一样，
    # 加一个很小的 epsilon 防止除 0。
    clip_coefficient = (
        max_l2_norm
        / (total_norm + 1e-6)
    )

    # norm 已经 <= max_norm 时不要放大梯度。
    if clip_coefficient < 1.0:
        with torch.no_grad():
            for grad in grads:
                grad.mul_(
                    clip_coefficient.to(
                        device=grad.device,
                        dtype=grad.dtype,
                    )
                )