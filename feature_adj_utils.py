import numpy as np
import torch
import os
import json
import logging
from typing import List, Dict, Optional
import tqdm
import gensim
from transformers import PreTrainedTokenizer
from graph_building_blocks.argument_set_punctuation_v4 import punctuations
from tokenization_dagn import arg_tokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
    relations = json.load(f)

def load_and_cache_examples(args,tokenizer,arg_tokenizer, evaluate=False, test=False):
    '''
    return:
        -dataset: TensorDataset
    '''
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # processors = {"race": RaceProcessor, "swag": SwagProcessor, "arc": ArcProcessor, "reclor": ReclorProcessor}
    # 返回一个key+value
    processor = processors['reclor']()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    #若之前已缓存，则直接读取，若未缓存，则先缓存再读取。
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples=examples,
            label_list=label_list,
            arg_tokenizer=arg_tokenizer,
            relations=relations,
            punctuations=punctuations,
            max_length=args.max_seq_length,
            tokenizer=tokenizer,
            max_ngram=5)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache



    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_passage_mask = torch.tensor([f.passage_mask for f in features], dtype=torch.long)
    all_question_mask = torch.tensor([f.question_mask for f in features], dtype=torch.long)
    all_argument_bpe_ids = torch.tensor([f.argument_bpe_ids for f in features], dtype=torch.long)
    all_domain_bpe_ids = torch.tensor([f.domain_bpe_ids for f in features], dtype=torch.long)
    all_punct_bpe_ids = torch.tensor([f.punct_bpe_ids for f in features], dtype=torch.long)
    all_keywordids = torch.tensor([f.keywordids for f in features], dtype=torch.long)
    all_keymask  = torch.tensor([f.keymask  for f in features], dtype=torch.long)
    all_SVO_ids = torch.tensor([f.SVO_ids for f in features], dtype=torch.long)  #[bsz,4,16]
    all_SVO_mask = torch.tensor([f.SVO_mask for f in features], dtype=torch.long)
    all_key_segid = torch.tensor([f.key_segid for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    # all_token_type_ids = torch.tensor([f.token_type_ids for f in features],dtype=torch.long)
    for f in features:    # list(len=4) of list(len=16) of list(max)
        for i in range(4):
            adj_of_end = f.adj_of_choice[i]  # list(len=16) of list(len=max)[[1,2,3],[1,2,3],[1,2,-1],[1,-1,-1],[-1,-1,-1]]
            pad = [-1]
            adj_of_end = [span+pad*(16-len(span)) for span in adj_of_end]
            f.adj_of_choice[i] = adj_of_end


    all_adj_SVO = torch.tensor([f.adj_of_choice for f in features], dtype=torch.long)  #[bsz,4,16,max_length]
    all_examle_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,all_passage_mask, all_question_mask,all_argument_bpe_ids,all_domain_bpe_ids,
                            all_punct_bpe_ids,all_keywordids,all_keymask,all_SVO_ids,all_SVO_mask,all_key_segid, all_label_ids, all_adj_SVO, all_examle_ids)
    return dataset



def SVOlist(svo_token_ids,tokenizer):
    '''
    input: svo_token_ids
    return:
        svo_list: list of length 16
        svo_mask: list of length 16
        adj: [[x,x,x,-1],[x,x,x,x]....]  list(len = 16) of list(len = max)
    '''
    length = len(svo_token_ids)
    svo_set = set()
    for i in range(length):
        svo_set.update(svo_token_ids[i][j] for j in range(len(svo_token_ids[i])))
    # print(svo_set)
    svo_list = list(svo_set)
    svo_list = sorted(svo_list,reverse=True)
    if len(svo_list) > 16:
        svo_list = svo_list[:16]
    padding = [0] * (16 - len(svo_list))
    svo_mask = [1] * len(svo_list)
    svo_list = svo_list + [tokenizer.pad_token_id] * (16 - len(svo_list))
    svo_list = svo_list[:16]
    svo_mask = svo_mask + padding
    svo_map = {}
    for i in range(length):
        for j in svo_token_ids[i]:
            if j in svo_map:
                svo_map[j].update(t for t in svo_token_ids[i])
            if not j in svo_map:
                svo_map[j] = set()
                svo_map[j].update(t for t in svo_token_ids[i])
    svo_map_sort = sorted(svo_map, reverse=True)
    adj = []
    for j in range(16):
        adj.append([])
    for i in range(len(svo_map_sort)):
        if i <= 15:
            adj_list = list(svo_map[svo_map_sort[i]])
            adj_ = set()
            for t in range(len(adj_list)):
                if adj_list[t] in svo_list:
                    adj_.update([svo_list.index(adj_list[t])])
            adj[i] = list(adj_)
    max_length = max(map(len, adj))
    A =[-1]
    adj = [span + A * (max_length-len(span)) for span in adj]
    return svo_list, svo_mask, adj

def keywordslist(keytokens_ids,tokenizer):
    keylist = []
    segid = []
    id = 0
    for i in keytokens_ids:
        for j in range(len(i)):
            keylist.append(i[j])
        id += len(i)
        segid.append(id)
    if len(keylist)>16:
        keylist = keylist[:16]
    padding = [0] * (16-len(keylist))
    mask = [1] *len(keylist)
    mask += padding
    keylist += [tokenizer.pad_token_id] * (16-len(keylist))
    if len(segid)>16:
        segid = segid[:16]
    padding_seg = [-1] * (16-len(segid))
    segid += padding_seg
    return keylist, mask, segid

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):  #构建一个样例 包含1context+4answers
    def __init__(self, example_id, input_ids, attention_mask,passage_mask,question_mask,
                 argument_bpe_ids, domain_bpe_ids,punct_bpe_ids,keywordids,keymask,
                 SVO_ids,SVO_mask,key_segid,adj_of_choice,label, token_type_ids):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.passage_mask = passage_mask
        self.question_mask = question_mask
        self.argument_bpe_ids = argument_bpe_ids
        self.domain_bpe_ids = domain_bpe_ids
        self.punct_bpe_ids = punct_bpe_ids
        self.keywordids = keywordids
        self.keymask = keymask
        self.SVO_ids =SVO_ids
        self.SVO_mask = SVO_mask
        self.key_segid = key_segid
        self.adj_of_choice =adj_of_choice
        self.label = label
        self.token_type_ids=token_type_ids

class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class ReclorProcessor(DataProcessor):
    """Processor for the ReClor data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label'] # for test set, there is no label. Just use 0 for convenience.
            if type == 'train':
                id_string = int(d['id_string'][6:])
            if type == 'test':
                id_string = int(d['id_string'][5:])
            if type == 'dev':
                id_string = int(d['id_string'][4:])
            examples.append(
                InputExample(
                    example_id = id_string,
                    question = question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label = label
                    )
                )
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list,
    arg_tokenizer,
    relations,
    punctuations ,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    max_ngram: int
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            stopwords = list(gensim.parsing.preprocessing.STOPWORDS) + punctuations
            inputs = arg_tokenizer(text_a, text_b, tokenizer, stopwords, relations, punctuations, max_ngram, max_length)
            choices_inputs.append(inputs)
        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        a_mask = [x["a_mask"] for x in choices_inputs]
        b_mask = [x["b_mask"] for x in choices_inputs]  # list[list]
        argument_bpe_ids = [x["argument_bpe_ids"] for x in choices_inputs]
        token_type_ids = [x["token_type_ids"] for x in choices_inputs]
        if isinstance(argument_bpe_ids[0], tuple):  # (argument_bpe_pattern_ids, argument_bpe_type_ids)
            arg_bpe_pattern_ids, arg_bpe_type_ids = [], []
            for choice_pattern, choice_type in argument_bpe_ids:
                assert (np.array(choice_pattern) > 0).tolist() == (
                            np.array(choice_type) > 0).tolist(), 'pattern: {}\ntype: {}'.format(
                    choice_pattern, choice_type)
                arg_bpe_pattern_ids.append(choice_pattern)
                arg_bpe_type_ids.append(choice_type)
            argument_bpe_ids = (arg_bpe_pattern_ids, arg_bpe_type_ids)
        domain_bpe_ids = [x["domain_bpe_ids"] for x in choices_inputs]
        punct_bpe_ids = [x["punct_bpe_ids"] for x in choices_inputs]
        keywordids = []
        keymask = []
        key_segid = []
        SVO_ids = []
        SVO_mask = []
        adj_of_choice = []
        for x in choices_inputs:
            keylist, mask, segid = keywordslist(x['keywords_ids'], tokenizer)
            keywordids.append(keylist)
            keymask.append(mask)
            key_segid.append(segid)
            SVO, SVOmask, adj = SVOlist(x['SVOlist'],tokenizer) #adj: list(len=16) of list(len =16)
            SVO_ids.append(SVO)
            SVO_mask.append(SVOmask)
            adj_of_choice.append(adj)  # list(len=4) of list(len=16) of list(max)

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                passage_mask=a_mask,
                question_mask=b_mask,
                argument_bpe_ids=argument_bpe_ids,
                domain_bpe_ids=domain_bpe_ids,
                punct_bpe_ids=punct_bpe_ids,
                keywordids=keywordids,
                keymask=keymask,
                SVO_ids=SVO_ids,
                SVO_mask=SVO_mask,
                key_segid=key_segid,
                adj_of_choice=adj_of_choice,
                label=label,
                token_type_ids=token_type_ids
            )
        )
    for f in features[:1]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


processors = {"reclor": ReclorProcessor}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race": 4, "swag": 4, "arc": 4, "reclor": 4}











