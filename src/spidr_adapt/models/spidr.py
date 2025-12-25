# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""SpidR model implementation."""

import copy
import logging
import math
from collections.abc import Iterable
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spidr_adapt.config import SpidRConfig
from spidr_adapt.data import SampleBatch, Tokenizer
from spidr_adapt.models.components import (
    LossWeightDefinition,
    SupervisedClassifier,
    Transformer,
    get_components,
    get_frame_level_cross_entropy_loss,
)
from spidr_adapt.models.metrics import perplexity

logger = logging.getLogger()


def exp_ema_scheduler(step: int, start_decay: float, timescale: float, threshold: float) -> float:
    decay = 1 - (1 - start_decay) * math.exp(-step / timescale)
    return decay if 1 - decay > threshold else 1


def init_teacher(
    student: Transformer, exclude_layers: Iterable[str], *, init_weights: bool = True
) -> tuple[Transformer, set[str]]:
    teacher = copy.deepcopy(student).float()
    if init_weights:
        teacher.apply(teacher.init_weights)
    teacher.eval()
    teacher.requires_grad_(requires_grad=False)
    teacher_exclude_layers: set[str] = set()
    for name, param in teacher.named_parameters():
        param.detach_()
        if any(name.startswith(ex) for ex in exclude_layers):
            teacher_exclude_layers.add(name)
    return teacher, teacher_exclude_layers


class SpidR(nn.Module):
    def __init__(self, cfg: SpidRConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = SpidRConfig()
        self.feature_extractor, self.feature_projection, self.student, self.heads, self.codebooks = get_components(cfg)
        self.teacher, self.teacher_exclude_layers = init_teacher(self.student, cfg.ema_exclude_layers)
        self.ema_scheduler = partial(
            exp_ema_scheduler,
            start_decay=cfg.ema_start_decay,
            timescale=cfg.ema_timescale,
            threshold=cfg.ema_threshold,
        )
        self.projection_dropout = nn.Dropout(cfg.encoder_projection_dropout)
        self.freeze_step = cfg.freeze_step
        self._extractor_frozen = False
        self.mask_embedding = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim))
        nn.init.uniform_(self.mask_embedding)
        self.current_step = nn.Buffer(torch.zeros(1, dtype=torch.int64))
        if cfg.supervised_every_step:
            self.prepare_for_supervised_training(cfg)
        self.supervised_layer = cfg.supervised_layer

    def train(self, mode: bool = True) -> "SpidR":  # noqa: FBT001, FBT002
        super().train(mode)
        self.teacher.eval()
        return self

    def prepare_for_supervised_training(self, cfg: SpidRConfig) -> None:
        self.tokenizers = {}
        self.supervised_classifiers = torch.nn.ModuleDict()
        for language in cfg.supervised_languages:
            self.initialize_supervised_language_components(language, cfg.encoder_embed_dim, cfg.num_supervised_layers)

    def initialize_supervised_language_components(
        self, language: str, encoder_embed_dim: int, num_supervised_layers: int
    ) -> None:
        logger.info("Initializing tokenizers and classifier heads for %s", language)
        self.tokenizers[language] = Tokenizer(language)
        self.supervised_classifiers[language] = SupervisedClassifier(
            encoder_embed_dim, self.tokenizers[language].vocab_size, num_supervised_layers
        )

    @property
    def num_codebooks(self) -> int:
        return len(self.codebooks)

    def freeze_extractor(self) -> None:
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.feature_projection.parameters():
            p.requires_grad = False
        self._extractor_frozen = True

    @torch.no_grad()
    def _inner_ema(self, decay: torch.Tensor) -> None:
        for (ema_n, ema_p), model_p in zip(self.teacher.named_parameters(), self.student.parameters(), strict=True):
            if ema_n in self.teacher_exclude_layers:
                ema_p.copy_(model_p)
            else:
                ema_p.lerp_(model_p, 1 - decay)
        for ema_b, model_b in zip(self.teacher.buffers(), self.student.buffers(), strict=True):
            ema_b.copy_(model_b)

    def update_ema(self, step: int) -> float:
        self.current_step.fill_(step)
        decay = self.ema_scheduler(step)
        if not self._extractor_frozen and step >= self.freeze_step:
            self.freeze_extractor()
        if 0.0 < decay < 1.0:
            self._inner_ema(torch.tensor(decay, device=self.current_step.device))
        return decay

    def get_intermediate_outputs(self, waveforms: Tensor, *, attention_mask: Tensor | None = None) -> list[Tensor]:
        x = self.feature_extractor(waveforms)
        x = self.feature_projection(x)
        return self.student.get_intermediate_outputs(x, attention_mask)

    def get_codebooks(
        self,
        waveform: Tensor,
        *,
        attention_mask: Tensor | None = None,
        onehot: bool = False,
    ) -> list[Tensor | None]:
        x = self.feature_extractor(waveform)
        x = self.feature_projection(x)
        preds: list[Tensor | None] = [None] * (len(self.student.layers) - self.num_codebooks)
        for i, y in enumerate(self.student.get_intermediate_outputs(x, attention_mask)[-self.num_codebooks :]):
            pred = self.heads[i](y).float().exp().squeeze()
            if onehot:
                pred = F.one_hot(pred.argmax(dim=-1), pred.size(-1))
            preds.append(pred)
        return preds

    def forward(
        self, batch: SampleBatch, *, loss_weights: LossWeightDefinition | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        feats = self.feature_extractor(batch.waveforms)
        feats = self.feature_projection(feats)
        x = feats.clone()
        x = self.projection_dropout(x)
        if batch.mask is not None:
            mask = batch.mask
            x = torch.where(batch.mask.unsqueeze(-1), self.mask_embedding.to(x.dtype).expand_as(x), x)
        else:
            mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        student_intermediate_outputs = self.student.get_intermediate_outputs(x, batch.attention_mask)
        mask_indices = torch.nonzero(mask, as_tuple=True)
        loss_weights = loss_weights or LossWeightDefinition()

        ssl_losses, target_ppl, pred_ppl = (
            torch.zeros(1, device=x.device),
            torch.zeros((), device=x.device),
            torch.zeros((), device=x.device),
        )
        if loss_weights.ssl > 0.0:
            with torch.no_grad():
                targets = self.teacher.get_intermediate_outputs(feats, batch.attention_mask)[-self.num_codebooks :]
                targets = [F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2)[mask_indices] for tl in targets]

            ssl_losses, target_ppl, pred_ppl = self.get_ssl_loss(
                student_intermediate_outputs, targets, mask_indices, x.device
            )

        supervision_loss = (
            self.get_supervision_loss(student_intermediate_outputs, batch.phonemes, batch.languages)
            if loss_weights.supervised > 0.0
            else torch.tensor(0.0, device=x.device, requires_grad=True)
        )

        return (
            (ssl_losses / self.num_codebooks),
            supervision_loss,
            {
                "target_ppl": target_ppl / self.num_codebooks,
                "pred_ppl": pred_ppl / self.num_codebooks,
            },
        )

    def get_ssl_loss(
        self,
        student_intermediate_outputs: list[Tensor],
        targets: list[Tensor],
        mask_indices: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        log_preds = [
            self.heads[i](y[mask_indices]) for i, y in enumerate(student_intermediate_outputs[-self.num_codebooks :])
        ]
        losses = torch.zeros(log_preds[0].shape[0], device=device)
        target_ppl, pred_ppl = torch.zeros((), device=device), torch.zeros((), device=device)
        for i, (log_pred, target) in enumerate(zip(log_preds, targets, strict=True)):
            onehot_target = self.codebooks[i](target)
            target_ppl += perplexity(onehot_target)
            pred_ppl += perplexity(log_pred.exp())
            losses += torch.sum(-onehot_target * log_pred, dim=-1)
        return losses, target_ppl, pred_ppl

    def get_supervision_loss(
        self, student_intermediate_outputs: list[Tensor], phonemes: Tensor, languages: list[str]
    ) -> Tensor:
        assert phonemes is not None, "Supervised training requires phoneme alignment input."
        languages_set = set(languages)
        assert len(languages_set) == 1, (
            "Supervised training requires a single language input. Ensure you are language batching."
        )
        language = languages_set.pop()
        assert language in self.supervised_classifiers, (
            "Language %s not initialized for supervised training.",
            language,
        )
        logits = self.supervised_classifiers[language](student_intermediate_outputs[self.supervised_layer - 1])
        return get_frame_level_cross_entropy_loss(logits, phonemes, self.tokenizers[language].ignore_id)
