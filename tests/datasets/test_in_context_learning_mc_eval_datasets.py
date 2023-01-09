# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import transformers

from composer.core import Evaluator
from composer.datasets.in_context_learning_multiple_choice_evaluation import get_mc_task_dataloader
from composer.loggers import InMemoryLogger
from composer.models.gpt2 import create_gpt2
from composer.trainer import Trainer
from tests.common import device, world_size


@pytest.mark.parametrize('dataset_uri', ['piqa_small.jsonz', 'hellaswag_small.jsonz'])
@world_size(1, 2)
@device('gpu', 'cpu')
@pytest.mark.parametrize('num_fewshot', [0, 1, 5])
def test_get_mc_task_dataloader(world_size, device, num_fewshot, dataset_uri, tiny_gpt2_tokenizer):
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    local_data = os.path.join(os.path.dirname(__file__), 'local_data')
    dataset_uri = f'{local_data}/{dataset_uri}'
    tokenizer = tiny_gpt2_tokenizer
    dl = get_mc_task_dataloader(dataset_uri,
                                tokenizer,
                                8,
                                max_seq_len=2048,
                                eos_tok_id=tokenizer.eos_token_id,
                                num_fewshot=num_fewshot,
                                preamble_string='',
                                example_delimiter='\n',
                                continuation_delimiter=': ')
    evaluator = Evaluator(label='lambada', dataloader=dl, metric_names=['InContextLearningMultipleChoiceAccuracy'])
    model = create_gpt2(use_pretrained=True, pretrained_model_name='EleutherAI/gpt-neo-125M')
    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)
    trainer.eval(eval_dataloader=evaluator)
    assert 'metrics/lambada/InContextLearningMultipleChoiceAccuracy' in in_memory_logger.data.keys()
    assert in_memory_logger.data['metrics/lambada/InContextLearningMultipleChoiceAccuracy'][0][1].item() > 0.5

