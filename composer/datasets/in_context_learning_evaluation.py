# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from composer.core import DataSpec
from composer.utils import dist
from composer.utils.file_helpers import get_file


class InContextLearningLMTaskDataset(Dataset):

    def __init__(
        self,
        dataset_uri: str,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        eos_tok_id: int,
        num_fewshot: int,
        preamble_string: str,
        example_delimiter: str,
        continuation_delimiter: str,
        destination_path: str = 'icl_lm_task.json',
    ):
        get_file(dataset_uri, destination_path, overwrite=True)
        dataset = load_dataset('json', data_files=destination_path, split='train', streaming=False)
        self.samples = list(
            dataset.map(lambda examples: {
                'continuation': examples['continuation'],
                'context': examples['context'],
            }))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_tok_id = eos_tok_id
        self.encoded_dataset = self.prep_examples(num_fewshot, preamble_string, example_delimiter,
                                                  continuation_delimiter)

    def prep_examples(self, num_fewshot, preamble_string, example_delimiter, continuation_delimiter):
        examples = []
        for sample_idx in range(len(self.samples)):
            encoded_example = {}

            encoded_example['preamble'] = preamble_string

            if num_fewshot > 0:
                allowable_indices = list(range(len(self.samples)))
                allowable_indices.remove(sample_idx)
                fewshot_idxs = random.sample(allowable_indices, num_fewshot)

                for fewshot_idx in fewshot_idxs:
                    ctxt, cont = self.samples[fewshot_idx]['context'], self.samples[fewshot_idx]['continuation']
                    if len(encoded_example['preamble']) > 0:
                        ctxt = f'{example_delimiter}{ctxt}'
                    encoded_example['preamble'] += f'{ctxt}{continuation_delimiter}{cont}'

            ctxt, cont = self.samples[sample_idx]['context'], self.samples[sample_idx]['continuation']
            if len(encoded_example['preamble']) > 0:
                ctxt = f'{example_delimiter}{ctxt}'

            cont = f'{continuation_delimiter}{cont}'

            encoded_example['context'] = self.tokenizer(ctxt)
            encoded_example['continuation'] = self.tokenizer(cont)
            encoded_example['preamble'] = self.tokenizer(encoded_example['preamble'])

            examples.append(encoded_example)

        return examples

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)

    def collate_fn(self, data):
        inputs = []
        continuation_indices = []
        for data_pair in data:
            preamble, context, continuation = (data_pair['preamble'], data_pair['context'], data_pair['continuation'])

            context_enc = preamble['input_ids'] + context['input_ids']
            continuation_enc = continuation['input_ids']
            continuation_span = torch.tensor(range(len(context_enc), len(context_enc) + len(continuation_enc)))

            inp = torch.tensor(
                (context_enc + continuation_enc)[-(self.max_seq_len + 1):],
                dtype=torch.long,
            )
            (inplen,) = inp.shape

            # pad length from seq to padding_length
            inp = torch.cat(
                [
                    inp,  # [seq]
                    torch.LongTensor((self.max_seq_len - inplen) * [self.eos_tok_id]),  # [padding_length - seq]
                ],
                dim=0,
            )

            inputs.append(inp)
            continuation_indices.append(continuation_span)

        return {
            'input_ids': torch.stack(inputs),
            'continuation_indices': continuation_indices,
            'mode': 'lm_task',
            'labels': torch.stack(inputs),
            'eos_tok_id': self.eos_tok_id
        }

    def get_num_samples_in_batch(self, batch) -> int:
        return batch['input_ids'].shape[0]

    def update_metric(self, metric, batch, output_logits, labels):
        metric.update(batch, output_logits, labels)


def get_lm_task_dataloader(
        dataset_uri: str,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_seq_len: int,
        eos_tok_id: int,
        num_fewshot: int,
        preamble_string: str,  # e.g. 'translate english to french:'
        example_delimiter: str,  # e.g. '\n'
        continuation_delimiter: str,  # e.g. ''
) -> DataSpec:
    dataset = InContextLearningLMTaskDataset(dataset_uri, tokenizer, max_seq_len, eos_tok_id, num_fewshot,
                                             preamble_string, example_delimiter, continuation_delimiter)
    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=True)
    print(f'Using microbatch size {batch_size}')
    return DataSpec(DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
    ),
                    device_transforms=None,
                    get_num_samples_in_batch=dataset.get_num_samples_in_batch)
