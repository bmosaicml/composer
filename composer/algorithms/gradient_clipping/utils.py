# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable

import torch


class OnlinePercentileEstimate:

    def __init__(
        self,
        tensor_shape,
        percentile=0.80,
        step=0.1,
        initial_estimate=0.2,
    ):
        self.step = torch.full(tensor_shape, step)
        self.step_up = torch.full(tensor_shape, 1.0 - percentile)
        self.step_down = torch.full(tensor_shape, percentile)
        self.percentile_estimate_tensor = torch.full(tensor_shape, initial_estimate)

        self._num_observations = 0

    def push(self, observation: torch.Tensor):
        # update percentile estimates
        step_up_tensor = -1 * (self.percentile_estimate_tensor > observation).float() * self.step * self.step_up
        step_down_tensor = (self.percentile_estimate_tensor < observation).float() * self.step * self.step_down
        self.percentile_estimate_tensor.add_(step_up_tensor).add_(step_down_tensor)

        # decrease step size
        step_decrement_factor = ((observation - self.percentile_estimate_tensor).abs() < self.step).float().mul_(1/2) \
            + ((observation - self.percentile_estimate_tensor).abs() >= self.step).float()
        self.step.mul_(step_decrement_factor)

        self._num_observations += 1

    @property
    def num_observations(self):
        return self._num_observations

    def query_percentile_threshold(self):
        return self.percentile_estimate_tensor


def unitwise_norm(tensor: torch.Tensor):
    """Implements unitwise norm as described in Brock et al, 2021.

    For 0D scalars of shape [], we trivially normalize with dim=0 which essentially returns the absolute value of the scalar.
    For 1D *.bias weights of shape [out_features], we normalize across entire vector -> dim=0.
    For 2D torch.nn.Linear weights of shape [out_features, in_features]: we normalize across in_features -> dim = 1
    For 4D torch.nn.Conv2d weights [out_channels, in_channels, kernel_height, kernel_width]:
        we normalize across [in_channels, kernel_height, kernel_width] -> dim = (1, 2, 3).
    If a 3D parameter were somehow in your model, we would normalize buy the last two dimensions -> dim = (1,2).

    Args:
        tensor (torch.Tensor): A parameter or gradient of the model.

    Returns:
        The appropriate L2 norm of the parameter or gradient as described above.
    """
    # 0D for scalars, 1D for bias vectors.
    if tensor.ndim <= 1:
        dim = 0
        keepdim = False
    # 2D corresponds to MLPs and 4D corresponds to ConvNets.
    else:
        dim = tuple(range(1, tensor.ndim))
        keepdim = True
    # L2 Norm.
    return torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=keepdim)
