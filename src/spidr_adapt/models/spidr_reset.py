# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""SpidR model implementation with codebook and prediction head reset."""

import itertools
import logging

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn import functional as F
from torch.optim import SGD

from spidr_adapt.config import SpidRWithResetConfig
from spidr_adapt.environment import pg_ddp
from spidr_adapt.metalearning.utils import sync_module_states
from spidr_adapt.models import SpidR
from spidr_adapt.models.metrics import perplexity

logger = logging.getLogger()


@torch.no_grad()
def reset_codebooks_and_heads(model: SpidR) -> None:
    for i, module in enumerate(model.codebooks):
        codebook_size, encoder_embed_dim = module.codebook.size()
        codebook = torch.randn(encoder_embed_dim, codebook_size, device=module.codebook.device).unsqueeze(0)
        module.codebook = F.instance_norm(codebook).transpose(1, 2).squeeze().float().contiguous()
        module.counts = torch.ones(codebook_size, dtype=torch.float32, device=module.counts.device).contiguous()
        model.heads[i][0].reset_parameters()
    if dist.is_initialized():
        to_ignore = {
            name
            for name, _ in itertools.chain(model.named_parameters(), model.named_buffers())
            if not name.startswith(("heads", "codebooks"))
        }
        sync_module_states(model, pg_ddp(), src=0, params_and_buffers_to_ignore=to_ignore)


@torch.autocast("cuda", enabled=False)
def adapt_head(head: nn.Sequential, inp_head: Tensor, labels: Tensor, *, lr: float = 5e-2, steps: int = 20) -> None:
    optimizer = SGD(head.parameters(), lr=lr)
    inp_head_nograd = inp_head.detach().float()
    for _ in range(steps):
        pred = head(inp_head_nograd)
        loss = F.nll_loss(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


class SpidRWithReset(SpidR):
    """Reset is done during the forward pass, with head adaptation."""

    def __init__(self, cfg: SpidRWithResetConfig) -> None:
        super().__init__(cfg)
        self.adapt_heads = cfg.adapt_heads
        self.task_interval = None

    def set_task_interval(self, task_interval: int) -> None:
        self.task_interval = task_interval

    def should_reset(self, step: int) -> bool:
        if self.task_interval is None or not self.training:
            return False
        return step % self.task_interval == 0

    def get_ssl_loss(
        self,
        student_intermediate_outputs: list[Tensor],
        targets: list[Tensor],
        mask_indices: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        should_reset = self.should_reset(self.current_step)
        interm = [y[mask_indices].float() for y in student_intermediate_outputs[-self.num_codebooks :]]

        losses = torch.zeros(interm[0].shape[0], device=device)
        target_ppl, pred_ppl = torch.zeros((), device=device), torch.zeros((), device=device)
        if should_reset:
            logger.info("Active Forgetting on codebook and heads at step %s", self.current_step.item())
            reset_codebooks_and_heads(self)
        for i, (y, target) in enumerate(zip(interm, targets, strict=True)):
            onehot_target = self.codebooks[i](target)
            if should_reset and self.adapt_heads:
                adapt_head(self.heads[i], y, onehot_target.argmax(-1))
            pred = self.heads[i](y).float()
            target_ppl += perplexity(onehot_target)
            pred_ppl += perplexity(pred.exp())
            losses += torch.sum(-onehot_target * pred, dim=-1)
        return losses, target_ppl, pred_ppl
