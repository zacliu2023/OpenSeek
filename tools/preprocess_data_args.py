# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

# from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset

from flagai.data.tokenizer import Tokenizer

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.initializer()

    def initializer(self):
        # Use Encoder class as a container for global data
        # Encoder.tokenizer = build_tokenizer(self.args)
        self.cache_dir = os.path.join(self.args.model_dir, self.args.model_name)
        try:
            Encoder.tokenizer = Tokenizer.from_pretrained(self.args.model_name,
                                                      cache_dir=self.cache_dir)
        except:
            from transformers import AutoTokenizer
            Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name,
                                                      cache_dir=self.cache_dir, trust_remote_code=True)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            text =f"<|extra_203|>{text}<|extra_204|>"
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                # sentence_ids = Encoder.tokenizer.tokenize(sentence)
                try:
                    sentence_ids = Encoder.tokenizer.encode_plus(sentence, None, max_length=None)['input_ids']

                    ## fill-in-middle
                    ## <|fim_begin|>pre<|fim_hole|>𝑓suf<|fim_end|>𝑓middle<|eos_token|>
                    import random
                    if self.args.fill_in_middle and random.randint(0, 99) < self.args.fill_in_middle_percentage:
                      fim_tokens =f"<|extra_200|><|extra_201|><|extra_202|>"
                      fim_tokens = Encoder.tokenizer.encode_plus(fim_tokens, None, max_length=None)['input_ids']
                      assert len(fim_tokens) == 3, "fim_tokens should be three tokens."
                      bos_token = sentence_ids[0]
                      eos_token = sentence_ids[-1]
                      words = sentence_ids[1:-1]
                      fim_begin_token = fim_tokens[0]
                      fim_hole_token = fim_tokens[1]
                      fim_end_token = fim_tokens[2]

                      if len(words) >= 3:
                        split1, split2 = sorted(random.sample(range(1, len(words)), 2))

                        prefix = words[:split1]
                        middle = words[split1:split2]
                        suffix = words[split2:]

                        sentence_ids_new = []
                        sentence_ids_new.append(bos_token)
                        sentence_ids_new.append(fim_begin_token)
                        sentence_ids_new.extend(prefix)
                        sentence_ids_new.append(fim_hole_token)
                        sentence_ids_new.extend(suffix)
                        sentence_ids_new.append(fim_end_token)
                        sentence_ids_new.extend(middle)
                        sentence_ids_new.append(eos_token)

                        assert len(sentence_ids_new) == len(sentence_ids) + 3, "sentence_ids_new should have three more tokens."
                        sentence_ids = sentence_ids_new

                    if len(sentence_ids) > 0:
                        doc_ids.append(sentence_ids)
                except:
                    continue
            ids[key] = doc_ids
        return ids, len(json_line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group.add_argument('--fill-in-middle', action='store_true',
                       help='Use Fill-in-Middle strategy.')
    group.add_argument('--fill-in-middle-percentage', type=int, default=10,
                       help='Percentage to use Fill-in-Middle strategy.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--model-name', type=str, required=True, default=None,
                       help='What model to use.')
    group.add_argument('--model-dir', type=str, required=True, default=None,
                       help='What model dir to use.')
    '''
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    '''

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk-size', type=int, required=True,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    '''
    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")
    '''

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args)
    tokenizer = encoder.tokenizer
    
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, args.chunk_size)
    #encoded_docs = map(encoder.encode, fin)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                               impl=args.dataset_impl,
                                               vocab_size=len(tokenizer.get_vocab()))

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items():
            if len(sentences) == 0:
                continue
            for sentence in sentences:
                # sentence = [151642] + sentence + [151643]
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

if __name__ == '__main__':
    main()
