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

from run import InputFeatures, convert_examples_to_features
from run_parser import get_identifiers, get_example
from utils import get_code_tokens, isUID, CodeDataset


# reuse tokenizer conversion strategy (tokenize -> ids -> pad)
# def convert_code_to_features(code, label, tokenizer, args):
#     """
#     将代码字符串转成 InputFeatures（与 run.py 的输入格式保持兼容）
#     """
#     # tokenize then clip to block
#     code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
#     source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
#     source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
#     padding_length = args.block_size - len(source_ids)
#     source_ids += [tokenizer.pad_token_id] * padding_length
#     return InputFeatures(source_tokens, source_ids, 0, label)

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
            if not all(x > self.args.block_size - 2 for x in position):
                filter_identifiers.append(identifier)
        return filter_identifiers

    def get_importance_score(self, example, code1, code2, words_list, variable_names, batch_size=16):
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

        # code2_tokens, _, _ = _tokenize(code2, self.tokenizer_tgt)

        # build examples: first original then masked variants
        new_examples = []
        # original
        new_code = self.tokenizer_tgt.tokenize(" ".join(words))
        new_examples.append(
            convert_examples_to_features(new_code, code2, example[1].item(), None, None, self.tokenizer_tgt, self.args,
                                         None))
        # masked variants
        for tokens in masked_token_list:
            new_examples.append(
                convert_examples_to_features(new_code, code2, example[1].item(), None, None, self.tokenizer_tgt,
                                             self.args, None))

        new_dataset = CodeDataset(new_examples)
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

    def attack_example(self, example, code_pair, code1, code2, rank, recoder):
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
        true_label = example[1].item()

        adv_code = ''
        temp_label = None

        # extract identifiers
        identifiers, orig_code_tokens = get_identifiers(code1, self.args.language_type)
        identifiers = [iden[0] for iden in identifiers]
        variable_names = self.filter_identifier(code1, identifiers)

        if not orig_label == true_label:
            print("skip for wrong predict")
            is_success = -4
            return code_pair, len(
                orig_code_tokens), adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        # tokenize processed code
        processed_code = " ".join(orig_code_tokens)
        words, sub_words, keys = None, None, None
        # We will pass words list as token list string to get_importance_score
        words_list = orig_code_tokens

        identifiers_2, code_tokens_2 = get_identifiers(code2, self.args.language_type)
        processed_code_2 = " ".join(code_tokens_2)
        code2 = " ".join(code2.split())
        words_2 = self.tokenizer_tgt.tokenize(code2)

        importance_score, replace_token_positions, names_positions_dict = self.get_importance_score(
            example, processed_code, words_2, words_list, variable_names, batch_size=self.args.eval_batch_size)

        if importance_score is None:
            return code_pair, len(
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

        final_code = copy.deepcopy(code1)
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
                recoder.write(rank, code_pair, temp_replace, true_label)
                temp_replace = " ".join(temp_replace.split())
                temp_code = self.tokenizer_tgt.tokenize(temp_replace)
                new_feature = convert_examples_to_features(temp_code, words_2, example[1].item(), None, None,
                                                           self.tokenizer_tgt, self.args, None)
                replace_examples.append(new_feature)

            if len(replace_examples) == 0:
                continue

            new_dataset = CodeDataset(replace_examples)
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
                    return code_pair, len(
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
        return code_pair, len(
            orig_code_tokens), adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
