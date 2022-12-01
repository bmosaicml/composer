# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Iterable, Optional, Union

import torch

from composer.algorithms.gradient_clipping.utils import OnlinePercentileEstimate, unitwise_norm


def apply_ope_gc(named_parameters: Union[torch.Tensor, Iterable[torch.Tensor]], percentile_threshold: float,
                 parameter_grad_percentile_estimates: Dict[str, OnlinePercentileEstimate], warmup_steps: int,
                 norm_type: str) -> Optional[dict]:
    """Clips all gradients in model based on ratio of gradient norms to parameter norms.
    Args:
        parameters (torch.Tensor or Iterable[torch.Tensor]): The parameters to of the
            model for whose gradients we will clip
        clipping_threshold (float, optional): The largest acceptable ratio between grad
            norms and parameter norms before clipping is done.
    """
    for name, param in named_parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()

        # Get clipped version of gradients.
        if norm_type == 'unitwise':
            grad_norm = unitwise_norm(grad)
        elif norm_type == 'param':
            grad_norm = torch.linalg.vector_norm(grad)
        else:
            raise ValueError(f"{norm_type} unsupported: must be one of ['unitwise', 'param']")

        if name not in parameter_grad_percentile_estimates:
            parameter_grad_percentile_estimates[name] = OnlinePercentileEstimate(grad_norm.shape,
                                                                                 percentile=percentile_threshold)

        parameter_grad_percentile_estimates[name].push(grad_norm)

        if parameter_grad_percentile_estimates[name].num_observations > warmup_steps:
            # Gradients whose norms are greater than weight_norm * clipping_threhsold are
            # scaled down by (weight_norm * clipping_threhsold) / grad_norm.
            max_norm = parameter_grad_percentile_estimates[name].query_percentile_threshold()
            clipped_grad_coeff = max_norm.div(grad_norm).nan_to_num_(nan=1.0).clamp(max=1.0)

            # Copy clipped gradients into param.grad attribute, so they can be accessed by
            # optimizer.
            grad.mul_(clipped_grad_coeff)

    return parameter_grad_percentile_estimates
