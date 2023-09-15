"""
Data process for Dataset

The generated dataset has output columns: ["input_ids", "attention_mask", "labels"],
and the data type of three columns is int64.

Columns：
    input_ids: the tokenized inputs, Tensor of shape :math:`(batch, seq_length)`.
    attention_mask: the mask indicating whether each position is a valid input and is not the added prompt,
                    Tensor of shape :math:`(batch, seq_length)`.
    labels: same as input_ids, Tensor of shape :math:`(batch, seq_length)`.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import numpy as np
import json

from mindspore.mindrecord import FileWriter
from mindformers.auto_class import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
max_seq_length = 1024 + 1024
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_tokens([])

def preprocess_data(input_file):
        model_inputs = []

        with open(input_file, 'r', encoding='utf-8') as fo:
            str = fo.read()
            examples = json.loads(str)

        for i in range(len(examples)):
            if examples[i]['question'] and examples[i]['answer']:
                query, answer = examples[i]['question'], examples[i]['answer']

                prompt = "问:{}, 答:{}".format(query,answer)
                '''
                a_ids = tokenizer.encode(prompt)
                b_ids = tokenizer.encode(answer)

                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                '''
                model_inputs.append(prompt)

        return model_inputs

def create_instance(ids, max_length=None):
    """A single sample instance for LM task."""

    tokens = tokenizer._convert_ids_to_tokens(ids)
    text = tokenizer._convert_tokens_to_string(tokens)
    output = tokenizer(text=text,
                        add_special_tokens=False,
                        max_length=max_length,
                        padding='max_length')
    return output


def write_instance_to_file(writer, instance):
    """write the instance to file"""
    print(instance)
    input_ids = instance["input_ids"]
    attention_mask = instance["attention_mask"]
    labels = instance["input_ids"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids)
    features["attention_mask"] = np.asarray(attention_mask)
    features["labels"] = np.asarray(labels)
    
    writer.write_raw_data([features])

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='.\data\\train.json',
                        help='Input raw text file. ')
    parser.add_argument("--output_file", type=str, default='.\data\\train.mindrecord',
                        help='Output MindRecord file. ')
    parser.add_argument("--num_splits", type=int, default=1,
                        help="The MindRecord file will be split into the number of partition. ")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length. ")
    parser.add_argument("--tokenizer_type", type=str, default="gpt2",
                        help="Tokenizer type, can be set to any tokenizer "
                             "if its relevant model supports prompt text classification. ")
    parser.add_argument("--data_columns", type=list, default=["input_ids", "attention_mask", "labels"],
                        help="The data columns which should be saved in mindrecord. This can refer used yaml file. ")

    args = parser.parse_args()

    input_file = args.input_file
    logging.info("***** Reading from input files *****")
    logging.info("Input File: %s", input_file)

    output_file = args.output_file
    logging.info("***** Writing to output files *****")
    logging.info("Output File: %s", output_file)

    writer = FileWriter(output_file, args.num_splits)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "attention_mask": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]}
                   }
    data_columns = args.data_columns
    need_del_keys = set(data_columns) - set(data_schema.keys())
    for need_del_key in need_del_keys:
        del data_schema[need_del_key]
    writer.add_schema(data_schema, "lm-schema")

    dataset_valid = preprocess_data(input_file)

    total_written = 0
    block_size = args.max_length
    total_ids = []
    for element in dataset_valid:
        total_ids += tokenizer.encode(element, add_special_tokens= False)
    total_length = len(total_ids)
    total_length = (total_length // block_size) * block_size
    print("total_length", total_length)
    for i in range(total_length // block_size):
        ids = total_ids[block_size*i:block_size*(i+1)]

        output = create_instance(ids, args.max_length)

        write_instance_to_file(writer, instance=output)
        total_written += 1

    
    logging.info("***** Reading from  %s *****", input_file)
    writer.commit()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

# python preprocess.py --input_file data\wikitext-2\wiki.train.tokens --output_file ./wikitext-2.train..mindrecord --max_length 1025