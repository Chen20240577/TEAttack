# -*- coding: utf-8 -*-
import sys
from copy import deepcopy

sys.path.append('../')
sys.path.append('../../../')
sys.path.append('../../../python_parser')

from load import load

from utils import get_identifier_posistions_from_code, get_masked_code_by_position
from utils import CodeDataset, GraphCodeDataset, CodePairDataset
from utils import is_valid_variable_name, get_code_tokens
from run_parser import get_identifiers, get_example


class Scanner:
    def __init__(self, args, model, tokenizer):
        self.target_model = model
        self.tokenizer = tokenizer
        self.device = args.device
        self.task = args.task
        self.model_type = args.model_type
        self.batch_size = args.batch_size
        self.language_type = args.language_type
        self.args = args

        if self.model_type == 'gcbert':
            self.code_max_len = args.data_flow_length + args.code_length
        else:
            self.code_max_len = args.block_size

        if self.model_type == 'bert' or self.model_type == 'gcbert' or self.model_type == 't5':
            self.front_token = tokenizer.cls_token
            self.behind_token = tokenizer.sep_token
        elif self.model_type == 'gpt2':
            self.front_token = tokenizer.eos_token
            self.behind_token = tokenizer.eos_token


    def Dataset(self, new_examples):
        if self.model_type == 'gcbert' and (self.task == 'authorship' or self.task == 'defect'):
            new_dataset = GraphCodeDataset(new_examples, self.args)
        elif self.model_type == 'gcbert' and self.task == 'clone':
            new_dataset = CodePairDataset(new_examples, self.args)
        else:
            new_dataset = CodeDataset(new_examples)

        return new_dataset


    def get_importance_score(self, code, label, variable_names, code_2):

        words, pre_words = load.pre_code(code)
        pre_words_2 = None

        if code_2 is not None:
            words_2, pre_words_2 = load.pre_code(code_2)

        # get identifier positions per name
        positions = get_identifier_posistions_from_code(words, variable_names)
        if len(positions) == 0:
            return None, None, None

        # compute masked token lists and replace positions (similar to wir)
        masked_token_list, replace_token_positions = get_masked_code_by_position(words, positions)

        # build examples: first original then masked variants
        new_examples = []
        # original


        new_examples.append(load.convert_example_to_features(self.args, pre_words, label, self.tokenizer, self.code_max_len, self.front_token, self.behind_token, pre_words_2))
        # masked variants
        for tokens in masked_token_list:
            new_examples.append(
                load.convert_example_to_features(self.args, " ".join(tokens), label, self.tokenizer, self.code_max_len, self.front_token, self.behind_token, pre_words_2))

        new_dataset = self.Dataset(new_examples)

        logits, preds = self.target_model.get_results(new_dataset, self.batch_size)
        # logits[0] is original prob vector
        orig_probs = logits[0]
        orig_label = preds[0]
        orig_prob_value = orig_probs[orig_label]  # predicted class prob

        importance_score = []
        for prob in logits[1:]:
            # prob is a vector of class probabilities after masking one token position
            importance_score.append(orig_prob_value - prob[orig_label])

        return importance_score, replace_token_positions, positions


    def filter_identifier(self, code, identifiers):
        code_token = get_code_tokens(code, self.tokenizer, self.language_type)
        filter_identifiers = []
        for identifier in identifiers:
            if is_valid_variable_name(identifier, self.language_type):
                position = []
                for index, token in enumerate(code_token):
                    if identifier == token:
                        position.append(index)
                if not all(x > self.code_max_len - 2 for x in position):
                    filter_identifiers.append(identifier)
        return filter_identifiers


    def scanner(self, code, label, code_2, identifiers):

        variable_names = self.filter_identifier(code, identifiers)

        importance_score, replace_token_positions, names_positions_dict \
            = self.get_importance_score(code, label, variable_names, code_2)

        if importance_score is None:
            return None, None

        # print(importance_score)
        # print(replace_token_positions)
        # print(names_positions_dict)

        token_pos_to_score_pos = {}
        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i

        # 计算Importance score，将所有出现的位置加起来
        names_to_importance_score = {}
        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score
            # print(names_to_importance_score)

        return names_positions_dict, names_to_importance_score





