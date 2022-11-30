# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ['OptimizerMonitor']


class OptimizerMonitor(Callback):
    """Computes and logs the L2 norm of gradients on the :attr:`.Event.AFTER_BATCH_END` event.

    L2 norms are calculated after the reduction of gradients across GPUs. This function iterates over the parameters of
    the model and may cause a reduction in throughput while training large models. In order to ensure the
    correctness of the norm, this function should be called after gradient unscaling in cases where gradients are scaled.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import GradMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[OptimizerMonitor()],
            ... )

    The L2 norms are logged by the :class:`.Logger` to the following keys as described below.

    +-----------------------------------+-------------------------------------------------------------+
    | Key                               | Logged data                                                 |
    +===================================+=============================================================+
    |                                   | L2 norm of the gradients of all parameters in the model     |
    | ``grad_l2_norm/step``             | on the :attr:`.Event.AFTER_TRAIN_BATCH` event.              |
    |                                   |                                                             |
    +-----------------------------------+-------------------------------------------------------------+
    |                                   | Layer-wise L2 norms if ``log_layer_grad_norms``             |
    | ``layer_grad_l2_norm/LAYER_NAME`` | is ``True``. Default: ``False``.                            |
    |                                   |                                                             |
    +-----------------------------------+-------------------------------------------------------------+
    |                                          | Layer-wise L2 norms of ``log_layer_grad_norms``      |
    | ``layer_grad_moment_l2_norm/LAYER_NAME`` | is ``True``. Default: ``False``.                     |
    |                                          |                                                      |
    +-----------------------------------+-------------------------------------------------------------+
    |                                   | Layer-wise L2 norms if ``log_layer_grad_norms``             |
    | ``layer_grad_l2_norm/LAYER_NAME`` | is ``True``. Default: ``False``.                            |
    |                                   |                                                             |
    +-----------------------------------+-------------------------------------------------------------+
    |                                   | Layer-wise L2 norms if ``log_layer_grad_norms``             |
    | ``layer_grad_l2_norm/LAYER_NAME`` | is ``True``. Default: ``False``.                            |
    |                                   |                                                             |
    +-----------------------------------+-------------------------------------------------------------+

    Args:
        log_layer_grad_norms (bool, optional): Whether to log the L2 normalization of each layer.
            Default: ``False``.
    """

    def __init__(self, log_layer_grad_norms: bool = False, log_optimizer_metrics: bool = False):
        self.log_layer_grad_norms = log_layer_grad_norms
        self.log_optimizer_metrics = log_optimizer_metrics

    def batch_end(self, state: State, logger: Logger):
        norm = 0.0
        layer_norms = {}
        optimizer_metrics = {}

        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                param_grad_norm = p.grad.detach().data.norm(2).item()  # type: ignore
                if self.log_layer_grad_norms:
                    layer_norms[f'layer_grad_l2_norm/{name}'] = param_grad_norm

                param_grad_norm = param_grad_norm**2
                norm += param_grad_norm
                metric_reporter = getattr(state.optimizers[0], 'report_per_parameter_metrics', None)
                if callable(metric_reporter) and self.log_optimizer_metrics:
                    optimizer_metrics = metric_reporter(p, name, optimizer_metrics)

        norm = norm**0.5
        logger.log_metrics({'grad_l2_norm/step': norm})
        if self.log_layer_grad_norms:
            logger.log_metrics(layer_norms)

        if self.log_optimizer_metrics:
            logger.log_metrics(optimizer_metrics)
