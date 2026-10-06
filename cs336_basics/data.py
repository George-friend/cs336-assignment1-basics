import numpy as np
import torch
from torch import Tensor


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """
    从一维 token dataset 中随机采样一个 language-modeling batch。

    Args:
        dataset:
            一维 numpy array，里面存储 token IDs。

        batch_size:
            一次采多少条序列。

        context_length:
            每条输入序列长度。

        device:
            "cpu"、"cuda:0" 等 PyTorch device。

    Returns:
        x:
            shape = (batch_size, context_length)

        y:
            shape = (batch_size, context_length)

        y 是 x 向后移动一个 token 的结果。
    """

    if dataset.ndim != 1:
        raise ValueError(
            "dataset must be a 1D numpy array."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if context_length <= 0:
        raise ValueError(
            "context_length must be positive."
        )

    if len(dataset) <= context_length:
        raise ValueError(
            "dataset must contain more tokens "
            "than context_length."
        )

    # --------------------------------------------------
    # 合法起点：
    #
    # start + context_length 必须仍然在 dataset 内
    #
    # 因为：
    #
    # x = dataset[start : start + context_length]
    #
    # y = dataset[start + 1 : start + context_length + 1]
    # --------------------------------------------------

    max_start = (
        len(dataset)
        - context_length
    )

    starting_indices = torch.randint(
        low=0,
        high=max_start,
        size=(batch_size,),
    )

    # --------------------------------------------------
    # 构造 input sequences
    # --------------------------------------------------

    x_numpy = np.stack(
        [
            dataset[
                start:
                start + context_length
            ]
            for start in starting_indices.tolist()
        ]
    )

    # --------------------------------------------------
    # 构造 next-token labels
    # --------------------------------------------------

    y_numpy = np.stack(
        [
            dataset[
                start + 1:
                start + context_length + 1
            ]
            for start in starting_indices.tolist()
        ]
    )

    # 作业要求 LongTensor
    x = torch.as_tensor(
        x_numpy,
        dtype=torch.long,
        device=device,
    )

    y = torch.as_tensor(
        y_numpy,
        dtype=torch.long,
        device=device,
    )

    return x, y