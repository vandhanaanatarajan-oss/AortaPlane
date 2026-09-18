import torch
from nnlandmark.training.nnUNetTrainer.project_specific.nnLandmark.nnLandmark_trainer import nnLandmark
from nnlandmark.training.lr_scheduler.polylr import PolyLRScheduler


def _get_encoder_params(network, stage_indices):
    """Yield parameters belonging to the given encoder stage indices."""
    params = []
    for name, p in network.named_parameters():
        if 'encoder.stages.' in name:
            parts = name.split('.')
            stage_idx = parts[parts.index('stages') + 1]
            if stage_idx in stage_indices:
                params.append((name, p))
    return params


def _get_all_encoder_params(network):
    return [(n, p) for n, p in network.named_parameters() if 'encoder.stages.' in n]


def _get_non_encoder_params(network):
    return [(n, p) for n, p in network.named_parameters() if not 'encoder.stages.' in n]


class nnLandmark_FrozenEncoder(nnLandmark):
    """Strategy A: freeze the entire pretrained encoder, train only decoder/head."""

    def configure_optimizers(self):
        encoder_params = _get_all_encoder_params(self.network)
        head_params = _get_non_encoder_params(self.network)

        frozen_count = 0
        for name, p in encoder_params:
            p.requires_grad = False
            frozen_count += 1

        self.print_to_log_file(f"[FrozenEncoder] Froze {frozen_count} encoder parameter tensors. "
                               f"Training {len(head_params)} decoder/head parameter tensors.")

        trainable_params = [p for _, p in head_params]
        optimizer = torch.optim.SGD(trainable_params, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnLandmark_PartialUnfreeze(nnLandmark):
    """Strategy B: freeze early encoder stages (0-2), train later stages (3-5) + decoder/head."""

    FROZEN_STAGES = {'0', '1', '2'}

    def configure_optimizers(self):
        frozen_params = _get_encoder_params(self.network, self.FROZEN_STAGES)
        trainable_encoder_params = [
            (n, p) for n, p in _get_all_encoder_params(self.network)
            if n.split('.')[n.split('.').index('stages') + 1] not in self.FROZEN_STAGES
        ]
        head_params = _get_non_encoder_params(self.network)

        for name, p in frozen_params:
            p.requires_grad = False

        self.print_to_log_file(f"[PartialUnfreeze] Froze {len(frozen_params)} params in encoder stages "
                               f"{sorted(self.FROZEN_STAGES)}. Training {len(trainable_encoder_params)} "
                               f"params in later encoder stages + {len(head_params)} decoder/head params.")

        trainable_params = [p for _, p in trainable_encoder_params] + [p for _, p in head_params]
        optimizer = torch.optim.SGD(trainable_params, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnLandmark_DifferentialLR(nnLandmark):
    """Strategy C: full fine-tune, but pretrained encoder gets a much smaller LR
    than the newly-initialized decoder/head (per supervisor's suggestion:
    1e-3 for new layers, 1e-4/1e-5 for pretrained encoder)."""

    ENCODER_LR = 1e-4
    HEAD_LR = 1e-3

    def configure_optimizers(self):
        encoder_params = [p for _, p in _get_all_encoder_params(self.network)]
        head_params = [p for _, p in _get_non_encoder_params(self.network)]

        self.print_to_log_file(f"[DifferentialLR] Encoder: {len(encoder_params)} params @ LR={self.ENCODER_LR}. "
                               f"Head: {len(head_params)} params @ LR={self.HEAD_LR}.")

        optimizer = torch.optim.SGD([
            {'params': encoder_params, 'lr': self.ENCODER_LR},
            {'params': head_params, 'lr': self.HEAD_LR},
        ], weight_decay=self.weight_decay, momentum=0.99, nesterov=True)

        # PolyLRScheduler decays based on optimizer.param_groups[0]['lr'] by default in most
        # nnU-Net versions -- we override step() to decay each group by its own initial LR ratio.
        lr_scheduler = _MultiGroupPolyLRScheduler(optimizer, [self.ENCODER_LR, self.HEAD_LR], self.num_epochs)
        return optimizer, lr_scheduler


class _MultiGroupPolyLRScheduler(object):
    """Applies poly LR decay independently to each param group, preserving
    each group's own initial LR ratio rather than forcing them to match group 0."""

    def __init__(self, optimizer, initial_lrs, max_steps, exponent=0.9):
        self.optimizer = optimizer
        self.initial_lrs = initial_lrs
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0

    def step(self, current_step=None):
        if current_step is None:
            current_step = self.ctr
            self.ctr += 1
        for i, group in enumerate(self.optimizer.param_groups):
            new_lr = self.initial_lrs[i] * (1 - current_step / self.max_steps) ** self.exponent
            group['lr'] = new_lr
