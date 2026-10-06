from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

import regex

from cs336_basics.tokenizer import PAT


def _merge_word(
    word: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    """
    把一个 word 中所有相邻的指定 pair 合并。

    例如：
    word = [1, 2, 1, 2]
    pair = (1, 2)
    new_token_id = 256

    结果：
    [256, 256]
    """
    new_word: list[int] = []
    i = 0

    while i < len(word):
        if (
            i + 1 < len(word)
            and word[i] == pair[0]
            and word[i + 1] == pair[1]
        ):
            new_word.append(new_token_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return new_word


def _count_pairs(
    word: list[int],
) -> Counter[tuple[int, int]]:
    """
    统计一个 word 中每种相邻 token pair 出现多少次。
    """
    return Counter(zip(word, word[1:]))


def _pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter[tuple[int, ...]]:
    """
    读取训练语料并执行 pre-tokenization。

    返回：
        pre-token 的 byte 序列 -> 出现次数

    special token 在 BPE 训练中作为硬边界：
    不参与 pair 统计，也不允许跨 special token merge。
    """

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:
        corpus = f.read()

    # --------------------------------------------------
    # 1. special token 作为硬边界切开语料
    # --------------------------------------------------

    if special_tokens:
        # 长 token 优先，避免有重叠 special token 时先匹配短的
        sorted_special_tokens = sorted(
            set(special_tokens),
            key=len,
            reverse=True,
        )

        special_pattern = "|".join(
            re.escape(token)
            for token in sorted_special_tokens
        )

        # 不使用捕获组，因此 special token 自己会被移除
        documents = re.split(
            special_pattern,
            corpus,
        )
    else:
        documents = [corpus]

    # --------------------------------------------------
    # 2. GPT 风格 pre-tokenization
    # --------------------------------------------------

    word_frequencies: Counter[tuple[int, ...]] = Counter()

    for document in documents:
        if not document:
            continue

        for match in regex.finditer(PAT, document):
            piece = match.group()

            # bytes 在 Python 中迭代出来就是 int
            # 例如：
            # b"cat" -> (99, 97, 116)
            byte_tuple = tuple(
                piece.encode("utf-8")
            )

            word_frequencies[byte_tuple] += 1

    return word_frequencies


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    """
    训练 byte-level BPE tokenizer。

    Args:
        input_path:
            训练文本路径。

        vocab_size:
            最终词表的最大大小。
            包含：
            - 256 个原始 byte token
            - special tokens
            - BPE merge 产生的新 token

        special_tokens:
            要加入 vocabulary 的特殊 token。

    Returns:
        vocab:
            token id -> bytes

        merges:
            按创建顺序保存的 BPE merge。
    """

    # ==================================================
    # 1. 建立最初的 byte vocabulary
    # ==================================================

    vocab: dict[int, bytes] = {
        i: bytes([i])
        for i in range(256)
    }

    # ==================================================
    # 2. 加入 special tokens
    # ==================================================

    for special_token in special_tokens:
        token_id = len(vocab)

        vocab[token_id] = special_token.encode(
            "utf-8"
        )

    if vocab_size < len(vocab):
        raise ValueError(
            "vocab_size must be at least "
            f"{len(vocab)} when using the given special tokens"
        )

    # ==================================================
    # 3. Pre-tokenization
    # ==================================================

    raw_word_frequencies = _pretokenize(
        input_path=input_path,
        special_tokens=special_tokens,
    )

    # 把每个 pre-token 变成 token-id list
    #
    # 例如：
    # b"cat"
    # ↓
    # [99, 97, 116]
    #
    # bytes 本身的值正好就是初始 token id。
    words: list[list[int]] = []
    word_frequencies: list[int] = []

    for byte_tuple, frequency in raw_word_frequencies.items():
        words.append(list(byte_tuple))
        word_frequencies.append(frequency)

    # ==================================================
    # 4. 建立初始 pair frequency
    # ==================================================

    # pair -> 整个 corpus 中出现多少次
    pair_frequencies: dict[
        tuple[int, int],
        int,
    ] = defaultdict(int)

    # pair -> 哪些 word 包含这个 pair
    #
    # 这是优化的关键：
    # merge 某个 pair 时，不必重新扫描全部 words。
    pair_to_words: dict[
        tuple[int, int],
        set[int],
    ] = defaultdict(set)

    for word_id, word in enumerate(words):
        frequency = word_frequencies[word_id]

        counts = _count_pairs(word)

        for pair, count in counts.items():
            pair_frequencies[pair] += (
                count * frequency
            )

            pair_to_words[pair].add(
                word_id
            )

    # ==================================================
    # 5. BPE merge
    # ==================================================

    merges: list[
        tuple[bytes, bytes]
    ] = []

    while (
        len(vocab) < vocab_size
        and pair_frequencies
    ):
        # ----------------------------------------------
        # 找出现次数最多的 pair
        #
        # 如果次数相同：
        # 选择 lexicographically greatest pair。
        #
        # 注意这里比较的是 token 对应的 bytes，
        # 而不是 token id。
        # ----------------------------------------------

        best_pair = max(
            pair_frequencies,
            key=lambda pair: (
                pair_frequencies[pair],
                vocab[pair[0]],
                vocab[pair[1]],
            ),
        )

        best_frequency = pair_frequencies[
            best_pair
        ]

        if best_frequency <= 0:
            break

        left_id, right_id = best_pair

        left_bytes = vocab[left_id]
        right_bytes = vocab[right_id]

        # ----------------------------------------------
        # 创建新的 token
        # ----------------------------------------------

        new_token_id = len(vocab)

        vocab[new_token_id] = (
            left_bytes + right_bytes
        )

        # merges 存的是 bytes，不是 ids
        merges.append(
            (
                left_bytes,
                right_bytes,
            )
        )

        # ----------------------------------------------
        # 只有包含 best_pair 的 words 会发生变化
        # ----------------------------------------------

        affected_word_ids = list(
            pair_to_words.get(
                best_pair,
                set(),
            )
        )

        for word_id in affected_word_ids:
            old_word = words[word_id]
            frequency = word_frequencies[
                word_id
            ]

            # ==========================================
            # A. 删除旧 word 对 pair statistics 的贡献
            # ==========================================

            old_pair_counts = _count_pairs(
                old_word
            )

            for pair, count in old_pair_counts.items():
                pair_frequencies[pair] -= (
                    count * frequency
                )

                pair_to_words[pair].discard(
                    word_id
                )

                # 清理已经不存在的 pair
                if pair_frequencies[pair] <= 0:
                    pair_frequencies.pop(
                        pair,
                        None,
                    )

                if (
                    pair in pair_to_words
                    and not pair_to_words[pair]
                ):
                    pair_to_words.pop(
                        pair,
                        None,
                    )

            # ==========================================
            # B. 真正执行 merge
            # ==========================================

            new_word = _merge_word(
                word=old_word,
                pair=best_pair,
                new_token_id=new_token_id,
            )

            words[word_id] = new_word

            # ==========================================
            # C. 加入新 word 的 pair statistics
            # ==========================================

            new_pair_counts = _count_pairs(
                new_word
            )

            for pair, count in new_pair_counts.items():
                pair_frequencies[pair] += (
                    count * frequency
                )

                pair_to_words[pair].add(
                    word_id
                )

    return vocab, merges