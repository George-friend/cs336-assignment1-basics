import math
from collections.abc import Iterable

import torch
from torch import Tensor


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer。

    Adam:
        m_t = beta1 * m_{t-1}
              + (1 - beta1) * g_t

        v_t = beta2 * v_{t-1}
              + (1 - beta2) * g_t^2

    然后做 bias correction，
    最后使用 decoupled weight decay。
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (
            0.9,
            0.999,
        ),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError(
                "Learning rate must be non-negative."
            )

        if eps < 0:
            raise ValueError(
                "eps must be non-negative."
            )

        if weight_decay < 0:
            raise ValueError(
                "weight_decay must be non-negative."
            )

        beta1, beta2 = betas

        if not 0 <= beta1 < 1:
            raise ValueError(
                "beta1 must be in [0, 1)."
            )

        if not 0 <= beta2 < 1:
            raise ValueError(
                "beta2 must be in [0, 1)."
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(
            params,
            defaults,
        )

    @torch.no_grad()
    def step(
        self,
        closure=None,
    ):
        """
        Perform one AdamW optimization step.
        """

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            beta1, beta2 = group["betas"]

            eps = group["eps"]

            weight_decay = (
                group["weight_decay"]
            )

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support "
                        "sparse gradients."
                    )

                state = self.state[
                    parameter
                ]

                # ==================================
                # 第一次看到这个参数时，
                # 初始化 optimizer state
                # ==================================

                if len(state) == 0:
                    state["step"] = 0

                    # 一阶矩
                    state["exp_avg"] = (
                        torch.zeros_like(
                            parameter
                        )
                    )

                    # 二阶矩
                    state["exp_avg_sq"] = (
                        torch.zeros_like(
                            parameter
                        )
                    )

                exp_avg = state[
                    "exp_avg"
                ]

                exp_avg_sq = state[
                    "exp_avg_sq"
                ]

                # ==================================
                # step
                # ==================================

                state["step"] += 1

                step = state["step"]

                # ==================================
                # AdamW:
                # decoupled weight decay
                #
                # θ <- θ * (1 - lr * lambda)
                # ==================================

                parameter.mul_(
                    1.0
                    - lr * weight_decay
                )

                # ==================================
                # 更新一阶矩
                #
                # m_t =
                # beta1 * m
                # +
                # (1-beta1) * grad
                # ==================================

                exp_avg.mul_(
                    beta1
                ).add_(
                    grad,
                    alpha=1.0 - beta1,
                )

                # ==================================
                # 更新二阶矩
                #
                # v_t =
                # beta2 * v
                # +
                # (1-beta2) * grad^2
                # ==================================

                exp_avg_sq.mul_(
                    beta2
                ).addcmul_(
                    grad,
                    grad,
                    value=1.0 - beta2,
                )

                # ==================================
                # Bias correction
                # ==================================

                bias_correction1 = (
                    1.0
                    - beta1 ** step
                )

                bias_correction2 = (
                    1.0
                    - beta2 ** step
                )

                # ==================================
                # PyTorch AdamW 风格更新
                # ==================================

                step_size = (
                    lr
                    / bias_correction1
                )

                bias_correction2_sqrt = (
                    math.sqrt(
                        bias_correction2
                    )
                )

                denominator = (
                    exp_avg_sq.sqrt()
                    / bias_correction2_sqrt
                )

                denominator.add_(eps)

                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-step_size,
                )

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Linear warmup + cosine decay。

    阶段 1:
        0 <= it < warmup_iters

        从 0 线性增加到 max_learning_rate

    阶段 2:
        warmup_iters <= it <= cosine_cycle_iters

        cosine decay:
        max_lr -> min_lr

    阶段 3:
        it > cosine_cycle_iters

        固定使用 min_learning_rate
    """

    # ==========================================
    # 1. Linear warmup
    # ==========================================

    if it < warmup_iters:
        return (
            max_learning_rate
            * it
            / warmup_iters
        )

    # ==========================================
    # 2. Cosine 已经结束
    # ==========================================

    if it > cosine_cycle_iters:
        return min_learning_rate

    # ==========================================
    # 3. Cosine decay
    # ==========================================

    decay_ratio = (
        (it - warmup_iters)
        /
        (
            cosine_cycle_iters
            - warmup_iters
        )
    )

    cosine_value = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * decay_ratio
            )
        )
    )

    learning_rate = (
        min_learning_rate
        +
        cosine_value
        * (
            max_learning_rate
            - min_learning_rate
        )
    )

    return learning_rate