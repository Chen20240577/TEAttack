# -*- coding: utf-8 -*-
"""
Accent -> classification attacker (identifier replacement).
核心思想：
- 计算每个 identifier 的 importance：用 masked / 替换后的概率下降（orig_prob - prob_after_mask）
- 根据 nearest_k 候选替换（若提供），依次尝试替换变量并调用模型判断是否能翻转标签
- 如果没有 nearest_k 提供，会从 id2token 中随机采样候选（保证形式类似 UID）
参考 wir_attacker.py、substitution.py、word_saliency.py 的实现风格。
"""

import sys

sys.path.append('../../../')
sys.path.append('../../../python_parser')
sys.path.append('../saved_models/')

import copy
import random

from run import InputFeatures, extract_dataflow
from run_parser import get_identifiers, get_example
from utils import get_code_tokens, isUID, GraphCodeDataset

from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_java import DFG_java
from parser_folder.DFG import DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,
                           tree_to_variable_index)

from tree_sitter import Language, Parser

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    # LANGUAGE = Language('parser/my-languages.so', lang)
    LANGUAGE = Language('../../../python_parser/parser_folder/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# reuse tokenizer conversion strategy (tokenize -> ids -> pad)
def convert_code_to_features(code, label, tokenizer, args):
    """
    将代码字符串转成 InputFeatures（与 run.py 的输入格式保持兼容）
    """
    # tokenize then clip to block
    parser = parsers["python"]
    code_tokens, dfg = extract_dataflow(code, parser, "python")
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length + args.data_flow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

    return InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, label)


class Accent_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, _token2idx, _idx2token, nearest_k_map=None) -> None:
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args
        self.nearest_k_map = nearest_k_map

    def filter_identifier(self, code, identifiers):
        """
        过滤 identifier（排除不合法、超长位置等），与 wir 的 filter 类似
        """
        code_token = get_code_tokens(code, self.tokenizer_tgt, self.args.language_type)
        filter_identifiers = []
        for identifier in identifiers:
            # only consider valid variable names (language-aware)
            if identifier is None:
                continue
            if identifier == '':
                continue
            # use provided helper if available
            # is_valid_variable_name may not exist in utils in your repo; if so, rely on simple checks:
            try:
                from utils import is_valid_variable_name
                valid = is_valid_variable_name(identifier, self.args.language_type)
            except Exception:
                # heuristic: must start with letter or underscore
                valid = (identifier[0].isalpha() or identifier[0] == '_')
            if not valid:
                continue
            position = [i for i, token in enumerate(code_token) if identifier == token]
            if len(position) == 0:
                continue
            if not all(x > self.args.data_flow_length + self.args.code_length - 2 for x in position):
                filter_identifiers.append(identifier)
        return filter_identifiers

    def get_importance_score(self, example, code, words_list, variable_names, batch_size=16):
        """
        计算每个 variable 的 importance score:
        - 用 mask 替换 variable，然后把 masked example 输入分类模型，衡量 orig_prob - prob_after 的差值
        - 返回 (importance_scores_list, replace_token_positions, names_positions_dict)
        与 WIR 的接口保持兼容（虽然实现不同）
        """
        # positions where identifiers occur in tokenized words_list
        # words_list expected as token list or string of tokens separated by space
        if isinstance(words_list, str):
            words = words_list.split()
        else:
            words = words_list

        # get identifier positions per name
        from utils import get_identifier_posistions_from_code, get_masked_code_by_position
        positions = get_identifier_posistions_from_code(words, variable_names)
        if len(positions) == 0:
            return None, None, None

        # compute masked token lists and replace positions (similar to wir)
        masked_token_list, replace_token_positions = get_masked_code_by_position(words, positions)

        # build examples: first original then masked variants
        new_examples = []
        # original
        new_examples.append(convert_code_to_features(" ".join(words), example[3].item(), self.tokenizer_tgt, self.args))
        # masked variants
        for tokens in masked_token_list:
            new_examples.append(
                convert_code_to_features(" ".join(tokens), example[3].item(), self.tokenizer_tgt, self.args))

        new_dataset = GraphCodeDataset(new_examples, self.args)
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
        # logits[0] is original prob vector
        orig_probs = logits[0]
        orig_label = preds[0]
        orig_prob_value = orig_probs[orig_label]  # predicted class prob

        importance_score = []
        for prob in logits[1:]:
            # prob is a vector of class probabilities after masking one token position
            importance_score.append(orig_prob_value - prob[orig_label])

        return importance_score, replace_token_positions, positions

    def attack_example(self, example, code, rank, recoder):
        """
        对单个 example 进行攻击（主流程）：
        - 计算变量 importance 排序
        - 对每个变量按 nearest_k 候选尝试替换（或从 idx2token 随机采样）
        - 若某一替换导致预测标签翻转则判定成功并返回 adv_code
        返回与 wir_attacker.wir_random_attack 保持一致的元组格式
        """
        # step 0: get original model outputs
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)
        true_label = example[3].item()

        adv_code = ''
        temp_label = None

        # extract identifiers
        identifiers, orig_code_tokens = get_identifiers(code, self.args.language_type)
        identifiers = [iden[0] for iden in identifiers]
        variable_names = self.filter_identifier(code, identifiers)

        if not orig_label == true_label:
            print("skip for wrong predict")
            is_success = -4
            return code, len(
                orig_code_tokens), adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        # tokenize processed code
        processed_code = " ".join(orig_code_tokens)
        words, sub_words, keys = None, None, None
        # We will pass words list as token list string to get_importance_score
        words_list = orig_code_tokens

        importance_score, replace_token_positions, names_positions_dict = self.get_importance_score(
            example, processed_code, words_list, variable_names, batch_size=self.args.eval_batch_size)

        if importance_score is None:
            return code, len(
                orig_code_tokens), adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None

        # map token_pos -> index in importance_score
        token_pos_to_score_pos = {}
        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i

        # aggregate importance per variable (sum across positions)
        names_to_importance_score = {}
        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                idx = token_pos_to_score_pos.get(token_pos, None)
                if idx is not None:
                    total_score += importance_score[idx]
            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)

        final_code = copy.deepcopy(code)
        nb_changed_var = 0
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        # iterate variables by importance
        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict.get(tgt_word, [])

            # get candidate substitutes
            candidates = None
            if self.nearest_k_map is not None:
                # nearest_k_map may be list-like: try to read mapping for this rank
                try:
                    # mapping could be list indexed by rank, or dict keyed by rank
                    if isinstance(self.nearest_k_map, list):
                        cand_dict = self.nearest_k_map[rank]
                    elif isinstance(self.nearest_k_map, dict):
                        cand_dict = self.nearest_k_map.get(rank, {})
                    else:
                        cand_dict = {}
                    candidates = cand_dict.get(tgt_word, [])
                except Exception:
                    candidates = []
            # fallback sampling from idx2token
            if not candidates:
                # sample up to 30 substitutes from idx2token that look like UIDs
                tmp_candidates = []
                tries = 0
                while len(tmp_candidates) < 30 and tries < 500:
                    tmp = random.choice(self.idx2token)
                    if isUID(tmp):
                        tmp_candidates.append(tmp)
                    tries += 1
                candidates = tmp_candidates

            if len(candidates) == 0:
                replaced_words[tgt_word] = tgt_word
                continue

            most_gap = 0.0
            candidate = None
            replace_examples = []
            substitute_list = []

            # create mutated examples (replace all occurrences)
            for substitute in candidates:
                substitute_list.append(substitute)
                temp_replace = get_example(final_code, tgt_word, substitute, self.args.language_type)
                recoder.write(rank, code, temp_replace, true_label)
                temp_replace = " ".join(temp_replace.split())
                new_feature = convert_code_to_features(temp_replace, example[3].item(), self.tokenizer_tgt, self.args)
                replace_examples.append(new_feature)

            if len(replace_examples) == 0:
                continue

            new_dataset = GraphCodeDataset(replace_examples, self.args)
            logits_subs, preds_subs = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert (len(logits_subs) == len(substitute_list))

            for idx_sub, temp_prob in enumerate(logits_subs):
                temp_label = preds_subs[idx_sub]
                if temp_label != orig_label:
                    # attack success
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(tgt_positions)
                    candidate = substitute_list[idx_sub]
                    replaced_words[tgt_word] = candidate
                    adv_code = get_example(final_code, tgt_word, candidate, self.args.language_type)

                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_prob,
                           temp_prob[orig_label]), flush=True)
                    return code, len(
                        orig_code_tokens), adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                else:
                    # no label change: measure prob gap
                    gap = current_prob - temp_prob[temp_label]
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[idx_sub]

            if most_gap > 0:
                # accept substitute that reduces predicted prob the most
                nb_changed_var += 1
                nb_changed_pos += len(tgt_positions)
                current_prob = current_prob - most_gap
                final_code = get_example(final_code, tgt_word, candidate, self.args.language_type)
                replaced_words[tgt_word] = candidate
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                      ('>>', tgt_word, candidate,
                       current_prob + most_gap,
                       current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

            adv_code = final_code

        # end for variables
        return code, len(
            orig_code_tokens), adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
