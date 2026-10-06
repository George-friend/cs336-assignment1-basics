import re
import regex

from typing import Iterable, Iterator


# GPT 风格的 pre-tokenization 正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # token_id -> bytes
        self.vocab = dict(vocab)

        # bytes -> token_id
        self.inverse_vocab = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        # BPE merge rules
        self.merges = merges

        # merge 的顺序就是优先级
        # rank 越小，优先级越高
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

        # 特殊 token
        self.special_tokens = special_tokens or []

        # special token 字符串 -> token id
        self.special_token_to_id = {}

        # 如果 special token 不在 vocab 中，就添加进去
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")

            if special_bytes not in self.inverse_vocab:
                new_id = max(self.vocab.keys(), default=-1) + 1

                self.vocab[new_id] = special_bytes
                self.inverse_vocab[special_bytes] = new_id

            self.special_token_to_id[special_token] = (
                self.inverse_vocab[special_bytes]
            )

    def _merge_pair(
        self,
        tokens: list[bytes],
        pair: tuple[bytes, bytes],
    ) -> list[bytes]:
        """
        将 tokens 中所有相邻的指定 pair 合并。
        """

        new_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append(
                    tokens[i] + tokens[i + 1]
                )
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def _encode_piece(self, piece: str) -> list[int]:
        """
        对一个 pre-tokenization 后的 piece 执行 BPE。
        """

        # str -> UTF-8 bytes
        text_bytes = piece.encode("utf-8")

        # 一开始，每个 byte 都是独立 token
        tokens = [
            bytes([byte])
            for byte in text_bytes
        ]

        # 持续执行 BPE merge
        while len(tokens) >= 2:
            best_pair = None
            best_rank = None

            # 检查当前所有相邻 pair
            for i in range(len(tokens) - 1):
                pair = (
                    tokens[i],
                    tokens[i + 1],
                )

                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]

                    if (
                        best_rank is None
                        or rank < best_rank
                    ):
                        best_pair = pair
                        best_rank = rank

            # 已经没有可以 merge 的 pair
            if best_pair is None:
                break

            tokens = self._merge_pair(
                tokens,
                best_pair,
            )

        # bytes token -> token id
        ids = [
            self.inverse_vocab[token]
            for token in tokens
        ]

        return ids

    def _encode_regular_text(
        self,
        text: str,
    ) -> list[int]:
        """
        普通文本编码：

        1. pre-tokenization
        2. 每一个 piece 单独做 BPE
        """

        ids = []

        pieces = regex.findall(
            PAT,
            text,
        )

        for piece in pieces:
            piece_ids = self._encode_piece(piece)
            ids.extend(piece_ids)

        return ids

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """
        字符串 -> token IDs。

        special token 保持整体；
        普通文本执行 pre-tokenization + BPE。
        """

        # 没有特殊 token
        if not self.special_tokens:
            return self._encode_regular_text(text)

        # 长的 special token 优先
        # 防止重叠 token 先匹配短的
        sorted_special_tokens = sorted(
            set(self.special_tokens),
            key=len,
            reverse=True,
        )

        pattern = "|".join(
            re.escape(token)
            for token in sorted_special_tokens
        )

        # () 保证 split 后 special token 自己不会消失
        parts = re.split(
            f"({pattern})",
            text,
        )

        ids = []

        for part in parts:
            if part == "":
                continue

            # special token：直接作为一个 token
            if part in self.special_token_to_id:
                ids.append(
                    self.special_token_to_id[part]
                )

            # 普通文本
            else:
                ids.extend(
                    self._encode_regular_text(part)
                )

        return ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        对一系列字符串进行流式编码。

        不把所有文本一次性加载到内存。
        """

        for text in iterable:
            yield from self.encode(text)

    def decode(
        self,
        ids: list[int],
    ) -> str:
        """
        token IDs -> 字符串。
        """

        token_bytes = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return token_bytes.decode(
            "utf-8",
            errors="replace",
        )