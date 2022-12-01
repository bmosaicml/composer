# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core gradient clipping classes and functions."""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Union

import torch

from composer.algorithms.gradient_clipping.online_percentile_gradient_clipping import apply_ope_gc
from composer.algorithms.gradient_clipping.utils import unitwise_norm
from composer.core import Algorithm, Event, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['GradientClipping', 'apply_gradient_clipping']


def apply_gradient_clipping(parameters: Union[torch.Tensor, Iterable[torch.Tensor]], clipping_type: str,
                            clipping_threshold: float) -> None:
    """Clips all gradients in model based on specified clipping_type.

    Args:
        parameters (torch.Tensor or Iterable[torch.Tensor]): The parameters to of the
            model for whose gradients we will clip
        clipping_type ('adaptive', 'norm', 'value'): String denoting which type of
            gradient clipping to do. The options are: 'norm', which clips the gradient norm
            and uses `torch.nn.utils.clip_grad_norm_`, 'value', which clips gradient at
            a specified value and uses `torch.nn.utils.clip_grad_value_`, and 'adaptive',
            which clips all gradients based on gradient norm:parameter norm ratio using
            composer.algorithms.gradient_clipping.gradient_clipping._apply_agc.
        clipping_threshold (float, optional): Specifies what value to clip the gradients
            to (for 'value'), what values to clip the gradient norms to (for 'norm'), and
            threshold by which if grad_norm / weight_norm is greater than this threshold then
            scale gradients by this threshold * (weight_norm / grad_norm) (for 'adaptive').
    """
    result = None
    if clipping_type == 'adaptive':
        _apply_agc(parameters, clipping_threshold=clipping_threshold)
    elif clipping_type == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=clipping_threshold)
    elif clipping_type == 'value':
        torch.nn.utils.clip_grad_value_(parameters, clip_value=clipping_threshold)
    else:
        raise ValueError(
            f"clipping_type must be 'adaptive', 'norm', 'online_percentile_estimate', or 'value' not {clipping_type} ")
    return result


def _apply_agc(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    clipping_threshold: float,
) -> None:
    """Clips all gradients in model based on ratio of gradient norms to parameter norms.
    Args:
        parameters (torch.Tensor or Iterable[torch.Tensor]): The parameters to of the
            model for whose gradients we will clip
        clipping_threshold (float, optional): The largest acceptable ratio between grad
            norms and parameter norms before clipping is done.
    """
    for param in parameters:
        if param.grad is None:
            continue

        # Detach weights and gradients, so the clipping operation is not added to
        # computational graph.
        weights = param.detach()
        grad = param.grad.detach()

        # Get clipped version of gradients.
        clipped_grad_coeff = _get_clipped_gradient_coeff(weights, grad, clipping_threshold=clipping_threshold)

        # Copy clipped gradients into param.grad attribute, so they can be accessed by
        # optimizer.
        grad.mul_(clipped_grad_coeff)


class GradientClipping(Algorithm):
    """Clips all gradients in model based on specified clipping_type.

    Runs on ``Event.AFTER_TRAIN_BATCH``.

    Example:
         .. testcode::

            from composer.algorithms import GradientClipping
            from composer.trainer import Trainer
            gc = GradientClipping(clipping_type='norm', clipping_threshold=0.1)
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[gc],
                optimizers=[optimizer]
            )

    Args:
        clipping_type ('adaptive', 'norm', 'value', 'online_percentile_estimate'): String denoting which type of
            gradient clipping to do. The options are: 'norm', which clips the gradient norm
            and uses `torch.nn.utils.clip_grad_norm_`, 'value', which clips gradient at
            a specified value and uses `torch.nn.utils.clip_grad_value_`, and 'adaptive',
            which clips all gradients based on gradient norm:parameter norm ratio using
            composer.algorithms.gradient_clipping.gradient_clipping._apply_agc.
        clipping_threshold (float, optional): Specifies what value to clip the gradients
            to (for 'value'), what values to clip the gradient norms to (for 'norm'), and
            threshold by which if grad_norm / weight_norm is greater than this threshold then
            scale gradients by this threshold * (weight_norm / grad_norm) (for 'adaptive').

    Raises:
        NotImplementedError: if deepspeed is enabled and clipping_type is not 'norm'.
        ValueError: if deepspeed is enabled and clipping_type is not 'norm'.
    """

    def __init__(self,
                 clipping_type: str,
                 clipping_threshold: float,
                 warmup_steps: Optional[int] = None,
                 norm_type: Optional[str] = None):
        self.clipping_type = clipping_type
        self.clipping_threshold = clipping_threshold
        if self.clipping_type == 'online_percentile_estimate':
            self.parameter_grad_percentile_estimates = {}
            self.warmup_steps = warmup_steps if warmup_steps else 0
            self.norm_type = norm_type if norm_type else 'unitwise'
            assert clipping_threshold >= 0 and clipping_threshold <= 1.0 and 'online percentile estimate requires the threshold be a percentile between 0 and 1'
        else:
            self.parameter_grad_percentile_estimates = None
            self.warmup_steps = None
            self.norm_type = None

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.INIT, Event.AFTER_TRAIN_BATCH]

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        if event == Event.INIT and state.deepspeed_config is not None:
            if self.clipping_type == 'norm':
                if self.clipping_threshold > 0:
                    state.deepspeed_config['gradient_clipping'] = self.clipping_threshold
                else:
                    raise ValueError(
                        f'Deepspeed only supports gradient clipping thresholds that are greater than zero, but the provided one is {self.clipping_threshold}'
                    )
            else:
                raise NotImplementedError(
                    f"Deepspeed only supports gradient clipping of type 'norm' not of type '{self.clipping_type}'")

        if event == Event.AFTER_TRAIN_BATCH and not state.deepspeed_enabled:
            if self.clipping_type == 'online_percentile_estimate':
                self.parameter_grad_percentile_estimates = apply_ope_gc(state.model.named_parameters(),
                                                                        self.clipping_threshold,
                                                                        self.parameter_grad_percentile_estimates,
                                                                        self.warmup_steps, self.norm_type)
            else:
                apply_gradient_clipping(parameters=state.model.parameters(),
                                        clipping_type=self.clipping_type,
                                        clipping_threshold=self.clipping_threshold)


def _get_clipped_gradient_coeff(weights: torch.Tensor, grad: torch.Tensor, clipping_threshold: float = 0.01):
    """Clips all gradients in model based on ratio of gradient norms to parameter norms.

    Gradients whose norms exceed

    .. math:: weight_norm * clipping_threshold

    are scaled down by

    .. math:: (weight_norm / grad_norm) * clipping_threshold.

    Args:
        weights (torch.Tensor): Tensor of weights (parameters) from the model.
        grad (torch.Tensor): Tensor of gradients of the loss with respect to the weights.
        clipping_threshold (float, optional): The largest acceptable ratio between grad
            norms and parameter norms before clipping is done.

    Return:
        clipped_grad_coeff (torch.Tensor): Coefficient of same shape as grad_norm equal to
            (weight_norm / grad_norm) * clipping_threshold for gradients whose norms
            that exceed weight_norm * clipping_threshold and one otherwise.
    """

    # Compute and clamp grad and weight norms.
    w_norm = unitwise_norm(weights)
    grad_norm = unitwise_norm(grad)

    # Gradients whose norms are greater than weight_norm * clipping_threhsold are
    # scaled down by (weight_norm * clipping_threhsold) / grad_norm.
    max_norm = w_norm.mul_(clipping_threshold)
    clipped_grad_coeff = max_norm.div_(grad_norm).nan_to_num_(nan=1.0).clamp_(max=1.0)

    return clipped_grad_coeff
