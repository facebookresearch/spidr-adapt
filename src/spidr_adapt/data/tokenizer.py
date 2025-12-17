# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tokenization for phoneme sequences."""

import json
from pathlib import Path

import torch


PHONEME_DICT_PATH = Path(__file__).parent / 'phoneme_dictionaries_vp19_train.json'


class Tokenizer:

    def __init__(self, language: str, with_blank: bool = False) -> None:  # noqa: FBT001, FBT002
        with Path.open(PHONEME_DICT_PATH) as file:
            self.phoneme_dict = json.load(file)
        self.token_to_id = self.get_phoneme_tokens(language)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.with_blank = with_blank

    def get_phoneme_tokens(self, language: str) -> dict[str, int]:
        assert language in self.phoneme_dict, (
            f"Missing language {language} in phoneme dict. Available languages are {self.phoneme_dict.keys()}"
        )
        phonemes = ["SIL", *self.phoneme_dict.get(language, "").split(" "), "UNK"]
        return {phone: idx for idx, phone in enumerate(phonemes)}

    @property
    def vocab_size(self) -> int:
        if not self.with_blank:
            return len(self.token_to_id) - 1
        return len(self.token_to_id)

    @property
    def silence_id(self) -> int:
        return self.token_to_id["SIL"]

    @property
    def ignore_id(self) -> int:
        # index to be used for unknown phonemes and padding
        return self.token_to_id["UNK"]

    def encode(self, phones: list[str] | str) -> torch.LongTensor:
        if isinstance(phones, str):
            phones = phones.split(" ")
        return torch.LongTensor([self.token_to_id[phone] for phone in phones])

    def decode(self, phones: torch.Tensor) -> list[str]:
        return [self.id_to_token[phone_id] for phone_id in phones.tolist()]


def get_tokenizer_for_lang(tokenizers: dict[str, Tokenizer], language: str) -> Tokenizer:
    if language not in tokenizers:
        tokenizers[language] = Tokenizer(language)
    return tokenizers[language]
