# -*- coding: utf-8 -*-
import copy
import random
import sys

import math
import pandas as pd

sys.path.append('../')
sys.path.append('../../../')
sys.path.append('../../../python_parser')

from load import load

from utils import get_identifier_posistions_from_code, get_masked_code_by_position, isUID, _tokenize
from utils import map_chromesome, select_parents, mutate, crossover
from utils import CodeDataset, GraphCodeDataset, CodePairDataset
from run_parser import get_identifiers, get_example

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def code_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]+', text)


def compute_cosine_similarity(original_code, adv_code, vectorizer=None):
    """计算两个代码之间的余弦相似度"""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(tokenizer=code_tokenizer, lowercase=False)
        # 用两个代码构建语料库
        corpus = [original_code, adv_code]
        vectorizer.fit(corpus)

    original_vector = vectorizer.transform([original_code])
    adv_vector = vectorizer.transform([adv_code])
    similarity = cosine_similarity(original_vector, adv_vector)[0][0]
    return similarity, vectorizer


class Attacker:
    def __init__(self, args, model, tokenizer, avg_sim, tfidf, nearest_k_map=None, id2token=None,substs=None) -> None:
        self.target_model = model
        self.tokenizer = tokenizer
        self.nearest_k_map = nearest_k_map
        self.device = args.device
        self.task = args.task
        self.model_type = args.model_type
        self.batch_size = args.batch_size
        self.language_type = args.language_type
        self.K = args.K
        self.args = args
        self.avg_sim = avg_sim
        self.thld = args.thld
        self.tfidf = tfidf
        self.id2token = id2token
        self.substs = substs

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

    def is_success(self, original_label, current_label):
        return original_label != current_label

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

        new_examples.append(
            load.convert_example_to_features(self.args, pre_words, label, self.tokenizer, self.code_max_len,
                                             self.front_token, self.behind_token, pre_words_2))
        # masked variants
        for tokens in masked_token_list:
            new_examples.append(
                load.convert_example_to_features(self.args, " ".join(tokens), label, self.tokenizer, self.code_max_len,
                                                 self.front_token, self.behind_token, pre_words_2))

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

    def get_candidates_for_variable(self, rank, tgt_word):
        """为单个变量获取候选词列表"""
        candidates = []

        # 从近义词映射获取候选词
        if self.nearest_k_map is not None:
            try:
                # 检查 nearest_k_map 的类型
                if isinstance(self.nearest_k_map, pd.DataFrame):
                    # 如果是DataFrame，通过rank(ID)查找对应的行
                    if rank < len(self.nearest_k_map):
                        row = self.nearest_k_map.iloc[rank]
                        cand_dict = row['nearest_k']  # 获取该代码的候选词字典
                        candidates = cand_dict.get(tgt_word, [])
                    else:
                        print(
                            f"Warning: rank {rank} out of range for nearest_k_map with length {len(self.nearest_k_map)}")

                elif isinstance(self.nearest_k_map, list):
                    # 如果是列表，直接通过索引访问
                    if rank < len(self.nearest_k_map):
                        cand_dict = self.nearest_k_map[rank]
                        candidates = cand_dict.get(tgt_word, [])

                elif isinstance(self.nearest_k_map, dict):
                    # 如果是字典，可能有不同的结构
                    cand_dict = self.nearest_k_map.get(rank, {})
                    candidates = cand_dict.get(tgt_word, [])

            except Exception as e:
                print(f"Error getting candidates for variable {tgt_word} at rank {rank}: {e}")
                candidates = []

        return candidates

    def has_untried_candidates(self, candidate_status):
        """检查是否还有未尝试的候选词"""
        for var_info in candidate_status.values():
            if var_info['untried_candidates']:
                return True
        return False

    def TEAA_greedy_attack(self, rank, example, code, label, code_2):
        logits, preds = self.target_model.get_results([example], self.batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        adv_code = ''
        temp_label = None

        # extract identifiers
        identifiers, orig_code_tokens = get_identifiers(code, self.args.language_type)
        variable_names = [iden[0] for iden in identifiers]
        # variable_names = self.filter_identifier(code, identifiers)

        if not orig_label == label:
            print("skip for wrong predict")
            is_success = -4
            return code, len(
                orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        importance_score, replace_token_positions, names_positions_dict = self.get_importance_score(
            code, label, variable_names, code_2)

        if importance_score is None:
            return code, len(
                orig_code_tokens), adv_code, label, orig_label, temp_label, -3, variable_names, None, None, None, None

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

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=self.args.reverse)
        # 改为从低到高

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
            candidates = self.get_candidates_for_variable(rank, tgt_word)
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
                temp_replace = " ".join(temp_replace.split())
                new_feature = load.convert_example_to_features(self.args, temp_replace, label, self.tokenizer,
                                                               self.code_max_len, self.front_token, self.behind_token,
                                                               code_2)
                replace_examples.append(new_feature)

            if len(replace_examples) == 0:
                continue

            new_dataset = self.Dataset(replace_examples)
            logits_subs, preds_subs = self.target_model.get_results(new_dataset, self.batch_size)
            assert (len(logits_subs) == len(substitute_list))

            for idx_sub, temp_prob in enumerate(logits_subs):
                temp_label = preds_subs[idx_sub]
                # if temp_label != orig_label:
                if self.is_success(temp_label,orig_label):
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
                        orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
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
            orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

    def TEAA_broad_attack(self, rank, example, code, label, code_2):
        logits, preds = self.target_model.get_results([example], self.batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        adv_code = ''
        temp_label = None

        # extract identifiers
        identifiers, orig_code_tokens = get_identifiers(code, self.args.language_type)
        variable_names = [iden[0] for iden in identifiers]

        if not orig_label == label:
            print("skip for wrong predict")
            is_success = -4
            return code, len(
                orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        importance_score, replace_token_positions, names_positions_dict = self.get_importance_score(
            code, label, variable_names, code_2)

        if importance_score is None:
            return code, len(
                orig_code_tokens), adv_code, label, orig_label, temp_label, -3, variable_names, None, None, None, None

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

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=self.args.reverse)

        # 初始化攻击状态
        current_code = copy.deepcopy(code)
        replaced_words = {}
        is_success = -1
        nb_changed_var = 0
        nb_changed_pos = 0
        temp_label = orig_label

        # 为每个变量初始化候选词和状态跟踪
        candidate_status = {}
        for tgt_word, importance in sorted_list_of_names:
            candidates = self.get_candidates_for_variable(rank, tgt_word)

            candidate_status[tgt_word] = {
                'all_candidates': candidates,
                'untried_candidates': candidates.copy(),  # 未尝试的候选词
                'tried_candidates': [],  # 已尝试的候选词
                'current_best_candidate': tgt_word,  # 当前最佳候选词
                'current_best_prob_drop': 0.0,
                'positions': names_positions_dict.get(tgt_word, [])
            }
            replaced_words[tgt_word] = tgt_word  # 初始化为自身

        batch = min(10, math.ceil(self.K / 2), self.batch_size)

        while self.has_untried_candidates(candidate_status) and is_success != 1:
            # 遍历每个变量（按重要性排序）
            for tgt_word, importance in sorted_list_of_names:
                var_status = candidate_status[tgt_word]

                # 如果该变量没有未尝试的候选词，跳过
                if not var_status['untried_candidates']:
                    continue

                # 批量处理：取一批未尝试的候选词（批量大小为self.batch_size）
                batch_candidates = []
                for _ in range(min(batch, len(var_status['untried_candidates']))):
                    candidate = var_status['untried_candidates'].pop(0)
                    # 跳过自身替换
                    if candidate != tgt_word:
                        batch_candidates.append(candidate)
                    var_status['tried_candidates'].append(candidate)

                # 如果没有有效的候选词，跳过当前变量
                if not batch_candidates:
                    continue

                # 批量生成替换后的代码
                batch_features = []
                candidate_replacements = {}

                for substitute in batch_candidates:
                    # 生成替换后的代码
                    temp_replace = get_example(current_code, tgt_word, substitute, self.args.language_type)
                    temp_replace_clean = " ".join(temp_replace.split())
                    candidate_replacements[substitute] = temp_replace_clean

                    # 创建特征
                    new_feature = load.convert_example_to_features(
                        self.args, temp_replace_clean, label, self.tokenizer, self.code_max_len,
                        self.front_token, self.behind_token, code_2
                    )
                    batch_features.append(new_feature)

                # 批量推理
                if batch_features:
                    new_dataset = self.Dataset(batch_features)
                    logits_sub_batch, preds_sub_batch = self.target_model.get_results(new_dataset, self.batch_size)

                    # 处理批量结果
                    batch_success = False
                    for i, (substitute, temp_prob, temp_label) in enumerate(
                            zip(batch_candidates, logits_sub_batch, preds_sub_batch)):
                        # 检查是否成功改变预测标签
                        # if temp_label != orig_label:
                        if self.is_success(temp_label,orig_label):
                            # 攻击成功！
                            is_success = 1
                            nb_changed_var += 1
                            nb_changed_pos += len(var_status['positions'])
                            replaced_words[tgt_word] = substitute
                            current_code = candidate_replacements[substitute]

                            print("%s SUC! %s => %s (%.5f => %.5f)" % \
                                  ('>>', tgt_word, substitute,
                                   current_prob,
                                   temp_prob[orig_label]), flush=True)

                            # 立即返回成功结果
                            return code, len(orig_code_tokens), current_code, label, orig_label, temp_label, \
                                is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                        else:
                            # 计算概率下降值
                            prob_drop = current_prob - temp_prob[orig_label]

                            # 如果这个候选词比当前最佳候选词效果更好，更新最佳候选词
                            if prob_drop > var_status['current_best_prob_drop']:
                                var_status['current_best_candidate'] = substitute
                                var_status['current_best_prob_drop'] = prob_drop

                            # 如果概率下降达到阈值，接受这个修改
                            if prob_drop > 0:
                                # 接受修改
                                current_prob = temp_prob[orig_label]
                                current_code = candidate_replacements[substitute]
                                replaced_words[tgt_word] = substitute
                                nb_changed_var += (replaced_words[tgt_word] != tgt_word)
                                nb_changed_pos += len(var_status['positions'])
                                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                                      ('>>', tgt_word, substitute,
                                       current_prob + prob_drop,
                                       temp_prob[orig_label]), flush=True)
                                # 应用修改后跳出当前批处理，继续下一个变量
                                break

            # 本轮循环结束后的处理：对每个变量应用当前最佳候选词（如果之前没有接受过修改）
            for tgt_word, importance in sorted_list_of_names:
                var_status = candidate_status[tgt_word]

                # 如果这个变量还没有被修改，但有有效的候选词
                if (replaced_words[tgt_word] == tgt_word and
                        var_status['current_best_prob_drop'] > 0 and
                        var_status['current_best_candidate'] != tgt_word):

                    # 应用当前最佳候选词
                    substitute = var_status['current_best_candidate']
                    temp_replace = get_example(current_code, tgt_word, substitute, self.args.language_type)

                    # 验证应用最佳候选词后的效果
                    temp_replace_clean = " ".join(temp_replace.split())
                    new_feature = load.convert_example_to_features(
                        self.args, temp_replace_clean, label, self.tokenizer, self.code_max_len,
                        self.front_token, self.behind_token, code_2
                    )

                    new_dataset = self.Dataset([new_feature])
                    logits_sub, preds_sub = self.target_model.get_results(new_dataset, self.batch_size)
                    temp_label = preds_sub[0]

                    # if temp_label != orig_label:
                    if self.is_success(temp_label, orig_label):
                        # 攻击成功！
                        is_success = 1
                        nb_changed_var += 1
                        nb_changed_pos += len(var_status['positions'])
                        replaced_words[tgt_word] = substitute
                        current_code = temp_replace
                        print("%s SUC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, substitute,
                               current_prob,
                               logits_sub[0][orig_label]), flush=True)

                        # 立即返回成功结果
                        return code, len(orig_code_tokens), current_code, label, orig_label, temp_label, \
                            is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                    else:
                        # 应用最佳候选词
                        current_code = temp_replace
                        replaced_words[tgt_word] = substitute
                        nb_changed_var += 1
                        nb_changed_pos += len(var_status['positions'])

        return code, len(orig_code_tokens), current_code, label, orig_label, temp_label, \
            is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

    def TEAA_subs_attack(self, rank, example, code, label, code_2, subs):
        logits, preds = self.target_model.get_results([example], self.batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        adv_code = ''
        temp_label = None

        # extract identifiers
        identifiers, orig_code_tokens = get_identifiers(code, self.args.language_type)
        variable_names = [iden[0] for iden in identifiers]
        # variable_names = self.filter_identifier(code, identifiers)

        if not orig_label == label:
            print("skip for wrong predict")
            is_success = -4
            return code, len(
                orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        importance_score, replace_token_positions, names_positions_dict = self.get_importance_score(
            code, label, variable_names, code_2)

        if importance_score is None:
            return code, len(
                orig_code_tokens), adv_code, label, orig_label, temp_label, -3, variable_names, None, None, None, None

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

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=self.args.reverse)
        # 改为从低到高

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
            candidates = subs

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
                temp_replace = " ".join(temp_replace.split())
                new_feature = load.convert_example_to_features(self.args, temp_replace, label, self.tokenizer,
                                                               self.code_max_len, self.front_token, self.behind_token,
                                                               code_2)
                replace_examples.append(new_feature)

            if len(replace_examples) == 0:
                continue

            new_dataset = self.Dataset(replace_examples)
            logits_subs, preds_subs = self.target_model.get_results(new_dataset, self.batch_size)
            assert (len(logits_subs) == len(substitute_list))

            for idx_sub, temp_prob in enumerate(logits_subs):
                temp_label = preds_subs[idx_sub]
                # if temp_label != orig_label:
                if self.is_success(temp_label, orig_label):
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
                        orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
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
            orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

    def MHM_thld_attack(self, code, _label, code_2, idx2token, _n_candi=30,
                        _max_iter=100, _prob_threshold=0.95, subs={}):
        orig_code = code
        identifiers, code_tokens = get_identifiers(code, self.args.language_type)
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer)
        raw_tokens = copy.deepcopy(words)
        variable_names = list(subs.keys())

        uid = get_identifier_posistions_from_code(words, variable_names)

        if len(uid) <= 0:
            # {'succ': True, 'tokens': code,
            #                             'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"],
            #                             "is_success": 1, "old_uid": old_uid, "score_info": res["old_prob"][0] - res["new_prob"][0],
            #                             "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos,
            #                             "replace_info": replace_info, "attack_type": "MHM-Origin"}
            return {'succ': None, 'tokens': None, 'raw_tokens': None, "prog_length": prog_length, "new_pred": _label,
                    "is_success": -1, "old_uid": "", "score_info": 0, "nb_changed_var": 0, "nb_changed_pos": None,
                    "replace_info": {}, "attack_type": "MHM-Origin"}

        variable_substitue_dict = {}
        for tgt_word in uid.keys():
            variable_substitue_dict[tgt_word] = subs[tgt_word]

        old_uids = {}
        old_uid = ""

        for iteration in range(1, 1 + _max_iter):
            res = self.__replaceUID_random(_tokens=code, code_2=code_2, idx2token=idx2token, _label=_label, _uid=uid,
                                           substitute_dict=variable_substitue_dict,
                                           _n_candi=_n_candi,
                                           _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")

            if res['status'].lower() in ['s', 'a']:
                if iteration == 1:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]
                flag = 0
                for k in old_uids.keys():
                    if res["old_uid"] == old_uids[k][-1]:
                        flag = 1
                        old_uids[k].append(res["new_uid"])
                        old_uid = k
                        break
                if flag == 0:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]

                code = res['tokens']
                uid[res['new_uid']] = uid.pop(res['old_uid'])
                variable_substitue_dict[res['new_uid']] = variable_substitue_dict.pop(res['old_uid'])
                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']

                if res['status'].lower() == 's':
                    replace_info = {}
                    nb_changed_pos = 0
                    for uid_ in old_uids.keys():
                        current_uid = old_uids[uid_][-1]
                        if current_uid not in uid:
                            print(f"Warning: {current_uid} not found in uid dictionary. Skipping.")
                            continue
                        replace_info[uid_] = current_uid
                        nb_changed_pos += len(uid[current_uid])

                    return {'succ': True, 'tokens': code,
                            'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"],
                            "is_success": 1, "old_uid": old_uid, "score_info": res["old_prob"][0] - res["new_prob"][0],
                            "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos,
                            "replace_info": replace_info, "attack_type": "MHM-Origin"}
        # 没有成功攻击的情况
        replace_info = {}
        nb_changed_pos = 0
        for uid_ in old_uids.keys():
            current_uid = old_uids[uid_][-1]
            if current_uid not in uid:
                print(f"Warning: {current_uid} not found in uid dictionary. Skipping.")
                continue
            replace_info[uid_] = current_uid
            nb_changed_pos += len(uid[current_uid])

        return {'succ': False, 'tokens': code, 'raw_tokens': None, "prog_length": prog_length,
                "new_pred": res["new_pred"] if 'res' in locals() else None, "is_success": -1,
                "old_uid": old_uid,
                "score_info": res["old_prob"][0] - res["new_prob"][0] if 'res' in locals() else 0,
                "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos,
                "replace_info": replace_info, "attack_type": "MHM-Origin"}

    def __replaceUID_random(self, _tokens, code_2, idx2token, _label=None, _uid={}, substitute_dict={},
                            _n_candi=30, _prob_threshold=0.95):

        selected_uid = random.sample(substitute_dict.keys(), 1)[0]  # 选择需要被替换的变量名

        # First, generate candidate set.
        # The transition probabilities of all candidate are the same.
        candi_token = [selected_uid]
        candi_tokens = [copy.deepcopy(_tokens)]
        candi_labels = [_label]
        for c in random.sample(idx2token, _n_candi):  # 选出_n_candi数量的候选.
            if isUID(c):  # 判断是否是变量名.
                candi_token.append(c)
                candi_tokens.append(copy.deepcopy(_tokens))
                candi_labels.append(_label)
                candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c, self.args.language_type)
                # for i in _uid[selected_uid]: # 依次进行替换.

        candi_texts = candi_tokens[1:]  # 从第二个开始，第一个是原始代码
        if candi_texts:  # 确保有候选代码需要计算
            similarities, threshold_flags = self.batch_compute_similarities(_tokens, candi_texts)
        else:
            similarities = []
            threshold_flags = []

        new_example = []
        for tmp_tokens in candi_tokens:
            tmp_code = tmp_tokens
            new_feature = load.convert_example_to_features(self.args, tmp_code, _label, self.tokenizer,
                                                           self.code_max_len, self.front_token, self.behind_token,
                                                           code_2)
            new_example.append(new_feature)
        new_dataset = self.Dataset(new_example)
        prob, pred = self.target_model.get_results(new_dataset, self.batch_size)

        best_diff = -1
        best_attack = None
        for i in range(len(candi_token)):  # Find a valid example
            # if pred[i] != _label:  # 如果有样本攻击成功
            #     return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
            #             "old_uid": selected_uid, "new_uid": candi_token[i],
            #             "old_prob": prob[0], "new_prob": prob[i],
            #             "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}

            # if pred[i] == _label:
            #     continue
            if not self.is_success(pred[i], _label):
                continue

            candi_idx = i - 1  # 调整为预计算结果的索引
            if threshold_flags[candi_idx]:
                if self.thld > 0.0:
                    print("**************** threshold is ok ****************")
                return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}

            # 使用预计算的相似度计算差异
            current_diff = abs(similarities[candi_idx] - self.avg_sim)

            if current_diff > best_diff:
                best_diff = current_diff
                best_attack = {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                               "old_uid": selected_uid, "new_uid": candi_token[i],
                               "old_prob": prob[0], "new_prob": prob[i],
                               "old_pred": pred[0], "new_pred": pred[i],
                               "nb_changed_pos": _tokens.count(selected_uid)}

        if best_attack is not None:
            print("**************** best_diff is ok ****************")
            return best_attack

        candi_idx = 0
        min_prob = 1.0

        for idx, a_prob in enumerate(prob[1:]):
            if a_prob[_label] < min_prob:
                candi_idx = idx + 1
                min_prob = a_prob[_label]

        # 找到Ground_truth对应的probability最小的那个mutant
        # At last, compute acceptance rate.
        alpha = (1 - prob[candi_idx][_label] + 1e-10) / (1 - prob[0][_label] + 1e-10)
        # 计算这个id对应的alpha值.
        if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
            return {"status": "r", "alpha": alpha, "tokens": candi_tokens[i],
                    "old_uid": selected_uid, "new_uid": candi_token[i],
                    "old_prob": prob[0], "new_prob": prob[i],
                    "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
        else:
            return {"status": "a", "alpha": alpha, "tokens": candi_tokens[i],
                    "old_uid": selected_uid, "new_uid": candi_token[i],
                    "old_prob": prob[0], "new_prob": prob[i],
                    "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}


    def batch_compute_similarities(self, original_code, candidate_codes):
        all_texts = [original_code] + candidate_codes
        # 批量向量化
        all_vectors = self.tfidf.transform(all_texts)
        # 获取原始代码向量（第一个）
        # original_vector = all_vectors[0:1]
        # 批量计算相似度
        similarities = cosine_similarity(all_vectors[0:1], all_vectors[1:])[0]
        # 批量计算阈值标签
        threshold_met = []
        for sim in similarities:
            if self.avg_sim == 0.0 or abs(sim - self.avg_sim) > self.thld * self.avg_sim:
                threshold_met.append(True)
            else:
                threshold_met.append(False)
        return similarities.tolist(), threshold_met

    def __printRes(self, _iter=None, _res=None, _prefix="  => "):
        if _res['status'].lower() == 's':  # Accepted & successful
            print("%s iter %d, SUCC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'r':  # Rejected
            print("%s iter %d, REJ. %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'a':  # Accepted
            print("%s iter %d, ACC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)

    def TEAA_num_attack(self, rank, example, code, label, code_2):

        logits, preds = self.target_model.get_results([example], self.batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        adv_code = ''
        temp_label = None

        # extract identifiers
        identifiers, orig_code_tokens = get_identifiers(code, self.args.language_type)
        variable_names = [iden[0] for iden in identifiers]

        if not orig_label == label:
            print("skip for wrong predict")
            is_success = -4
            return code, len(
                orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        importance_score, replace_token_positions, names_positions_dict = self.get_importance_score(
            code, label, variable_names, code_2)

        if importance_score is None:
            return code, len(
                orig_code_tokens), adv_code, label, orig_label, temp_label, -3, variable_names, None, None, None, None

        # 【新增】计算每个标识符的出现次数
        names_to_occurrence_count = {}
        for name, positions in names_positions_dict.items():
            # positions 列表的长度就是该标识符在代码中出现的次数
            names_to_occurrence_count[name] = len(positions)

        # 【修改】按照出现次数进行排序，出现次数多的排前面
        sorted_list_of_names = sorted(names_to_occurrence_count.items(), key=lambda x: x[1], reverse=True)

        final_code = copy.deepcopy(code)
        nb_changed_var = 0
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        # iterate variables by occurrence count (从出现次数最多的开始)
        for name_and_count in sorted_list_of_names:
            tgt_word = name_and_count[0]
            tgt_positions = names_positions_dict.get(tgt_word, [])
            occurrence_count = name_and_count[1]  # 标识符的出现次数

            # get candidate substitutes
            candidates = self.get_candidates_for_variable(rank, tgt_word)
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
                temp_replace = " ".join(temp_replace.split())
                new_feature = load.convert_example_to_features(self.args, temp_replace, label, self.tokenizer,
                                                               self.code_max_len, self.front_token, self.behind_token,
                                                               code_2)
                replace_examples.append(new_feature)

            if len(replace_examples) == 0:
                continue

            new_dataset = self.Dataset(replace_examples)
            logits_subs, preds_subs = self.target_model.get_results(new_dataset, self.batch_size)
            assert (len(logits_subs) == len(substitute_list))

            for idx_sub, temp_prob in enumerate(logits_subs):
                temp_label = preds_subs[idx_sub]
                if self.is_success(temp_label, orig_label):
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
                        orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, names_to_occurrence_count, nb_changed_var, nb_changed_pos, replaced_words
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
            orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, names_to_occurrence_count, nb_changed_var, nb_changed_pos, replaced_words

    def compute_fitness(self, chromesome, orig_prob, orig_label, label, code, code_2):
        # 计算fitness function.
        # words + chromesome + orig_label + current_prob
        temp_code = map_chromesome(chromesome, code, self.args.language_type)
        new_feature = load.convert_example_to_features(self.args, temp_code, label, self.tokenizer,
                                                               self.code_max_len, self.front_token, self.behind_token,
                                                               code_2)
        new_dataset = self.Dataset([new_feature])
        new_logits, preds = self.target_model.get_results(new_dataset, self.batch_size)
        # 计算fitness function
        fitness_value = orig_prob - new_logits[0][orig_label]
        return fitness_value, preds[0]

    def Alert_subs_get(self, index, tgt_word):
        if self.args.subs == 'random':
            # 随机生成候选词列表
            _n_candi = self.args.K  # 随机采样的数量
            candidates = []

            # 从 id2token 中随机采样生成候选列表
            # 过滤出有效的标识符
            valid_identifiers = [c for c in self.id2token if isUID(c) and c != tgt_word]

            if len(valid_identifiers) >= _n_candi:
                # 如果有足够多的有效标识符，直接随机采样
                candidates = random.sample(valid_identifiers, _n_candi)
            else:
                # 如果有效标识符不足，先添加所有有效标识符
                candidates = valid_identifiers.copy()
                # 如果还不够，从所有标识符中补充（不重复）
                remaining_needed = _n_candi - len(candidates)
                if remaining_needed > 0 :
                    # 从所有标识符中随机采样，排除已选的和原词
                    all_candidates = [c for c in self.id2token if c != tgt_word and c not in candidates]
                    if len(all_candidates) >= remaining_needed:
                        additional = random.sample(all_candidates, remaining_needed)
                        candidates.extend(additional)

            return candidates
        elif self.args.subs == 'word2vec' or self.args.subs == 'wordnet':
            return self.get_candidates_for_variable(index, tgt_word)
        elif self.args.subs == 'model':
            if tgt_word in self.substs[index]:
                return self.substs[index][tgt_word]
        else:
            return []

    def Alert_GA_thld(self, rank, example, code, label, code_2, initial_replace=None):
        logits, preds = self.target_model.get_results([example], self.batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = label
        adv_code = ''
        temp_label = None

        identifiers, orig_code_tokens = get_identifiers(code, self.args.language_type)
        prog_length = len(orig_code_tokens)
        processed_code = " ".join(orig_code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer)
        # 这里经过了小写处理..
        variable_names = [iden[0] for iden in identifiers]

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        final_code = copy.deepcopy(code)
        nb_changed_var = 0  # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1

        # 将原始代码转换为分词列表字符串形式，用于相似度计算
        original_tokens_str = " ".join(orig_code_tokens) if isinstance(orig_code_tokens, list) else orig_code_tokens
        # 存储所有攻击成功的突变体（满足阈值或不满足阈值）
        successful_mutants = []

        # 我们可以先生成所有的substitues
        variable_substitue_dict = {}

        for tgt_word in names_positions_dict.keys():
            candidates= self.Alert_subs_get(rank, tgt_word)
            if candidates == None:
                continue
            if len(candidates) == 0:
                continue
            variable_substitue_dict[tgt_word] = candidates

        if len(variable_substitue_dict) == 0:
            # 没找到替换词的情况，跳过
            is_success = -2
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        fitness_values = []
        base_chromesome = {word: word for word in variable_substitue_dict.keys()}
        population = [base_chromesome]
        # 关于chromesome的定义: {tgt_word: candidate, tgt_word_2: candidate_2, ...}
        for tgt_word in variable_substitue_dict.keys():
            # 这里进行初始化
            if initial_replace is None:
                # 对于每个variable: 选择"影响最大"的substitues
                replace_examples = []
                substitute_list = []

                current_prob = max(orig_prob)
                most_gap = 0.0
                initial_candidate = tgt_word
                tgt_positions = names_positions_dict[tgt_word]

                # 原来是随机选择的，现在要找到改变最大的.
                for substitute in variable_substitue_dict[tgt_word]:
                    substitute_list.append(substitute)
                    # 记录下这次换的是哪个substitue
                    temp_replace = get_example(final_code, tgt_word, substitute, self.args.language_type)
                    temp_replace = " ".join(temp_replace.split())
                    new_feature = load.convert_example_to_features(self.args, temp_replace, label, self.tokenizer,
                                                                   self.code_max_len, self.front_token,
                                                                   self.behind_token,
                                                                   code_2)
                    replace_examples.append(new_feature)

                if len(replace_examples) == 0:
                    # 并没有生成新的mutants，直接跳去下一个token
                    continue

                new_dataset = self.Dataset(replace_examples)
                # 3. 将他们转化成features
                logits_subs, preds_subs = self.target_model.get_results(new_dataset, self.args.batch_size)
                assert (len(logits_subs) == len(substitute_list))

                _the_best_candidate = -1
                for index, temp_prob in enumerate(logits_subs):
                    temp_label = preds_subs[index]
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        _the_best_candidate = index
                if _the_best_candidate == -1:
                    initial_candidate = tgt_word
                else:
                    initial_candidate = substitute_list[_the_best_candidate]
            else:
                initial_candidate = initial_replace[tgt_word]

            temp_chromesome = copy.deepcopy(base_chromesome)
            temp_chromesome[tgt_word] = initial_candidate
            population.append(temp_chromesome)
            temp_fitness, temp_label = self.compute_fitness(
                temp_chromesome, max(orig_prob), orig_label, true_label, final_code, code_2)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7

        max_iter = max(5 * len(population), 10)
        # 这里的超参数还是的调试一下.

        for i in range(max_iter):
            _temp_mutants = []
            for j in range(self.args.batch_size):
                p = random.random()
                chromesome_1, index_1, chromesome_2, index_2 = select_parents(population)
                if p < cross_probability:  # 进行crossover
                    if chromesome_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                        continue
                    child_1, child_2 = crossover(chromesome_1, chromesome_2)
                    if child_1 == chromesome_1 or child_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                else:  # 进行mutates
                    child_1 = mutate(chromesome_1, variable_substitue_dict)
                _temp_mutants.append(child_1)

            # compute fitness in batch
            feature_list = []
            for mutant in _temp_mutants:
                _temp_code = map_chromesome(mutant, final_code, self.args.language_type)
                _tmp_feature = load.convert_example_to_features(self.args, _temp_code, label, self.tokenizer,
                                                                self.code_max_len, self.front_token, self.behind_token,
                                                                code_2)
                feature_list.append(_tmp_feature)
            if len(feature_list) == 0:
                continue
            new_dataset = self.Dataset(feature_list)
            mutate_logits, mutate_preds = self.target_model.get_results(new_dataset, self.args.batch_size)
            mutate_fitness_values = []

            # 收集所有攻击成功的突变体
            current_batch_successful = []
            for index, logits in enumerate(mutate_logits):
                if mutate_preds[index] != orig_label:
                    # 攻击成功，记录突变体信息
                    mutant_code = map_chromesome(_temp_mutants[index], final_code, self.args.language_type)
                    current_batch_successful.append({
                        'index': index,
                        'chromesome': _temp_mutants[index],
                        'code': mutant_code,
                        'pred': mutate_preds[index],
                        'logits': logits
                    })

            # 处理当前批次的攻击成功突变体
            if current_batch_successful:
                # 批量计算相似度
                successful_codes = [mutant['code'] for mutant in current_batch_successful]
                similarities, threshold_flags = self.batch_compute_similarities(
                    original_tokens_str,
                    successful_codes
                )

                for j, mutant in enumerate(current_batch_successful):
                    if threshold_flags[j]:
                        # 满足阈值条件，返回这个突变体
                        chromesome = mutant['chromesome']
                        adv_code = mutant['code']
                        temp_label = mutant['pred']
                        is_success = 1

                        # 计算改变的变量数和位置
                        nb_changed_var = 0
                        nb_changed_pos = 0
                        for old_word in chromesome.keys():
                            if old_word != chromesome[old_word]:
                                nb_changed_var += 1
                                nb_changed_pos += len(names_positions_dict.get(old_word, []))

                        if self.thld > 0.0:
                            print("**************** threshold is ok ****************")
                        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, nb_changed_var, nb_changed_pos, chromesome

                # 如果没有满足阈值条件的攻击样本，记录所有攻击成功的突变体
                for j, mutant in enumerate(current_batch_successful):
                    successful_mutants.append({
                        'chromesome': mutant['chromesome'],
                        'code': mutant['code'],
                        'pred': mutant['pred'],
                        'logits': mutant['logits'],
                        'similarity': similarities[j]
                    })

            # 如果没有立即返回，继续计算适应度并更新种群
            for index, logits in enumerate(mutate_logits):
                _tmp_fitness = max(orig_prob) - logits[orig_label]
                mutate_fitness_values.append(_tmp_fitness)

            # 现在进行替换.
            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    # 替换.
                    min_index = fitness_values.index(min_value)
                    population[min_index] = _temp_mutants[index]
                    fitness_values[min_index] = fitness_value

            # 迭代结束后，如果找到了攻击成功的突变体但没有满足阈值的
        if successful_mutants:
            # 选择与平均相似度差异最大的攻击样本
            best_diff = -1
            best_mutant = None

            for mutant in successful_mutants:
                current_diff = abs(mutant['similarity'] - self.avg_sim)
                if current_diff > best_diff:
                    best_diff = current_diff
                    best_mutant = mutant

            if best_mutant is not None:
                chromesome = best_mutant['chromesome']
                adv_code = best_mutant['code']
                temp_label = best_mutant['pred']
                is_success = 1

                # 计算改变的变量数和位置
                nb_changed_var = 0
                nb_changed_pos = 0
                for old_word in chromesome.keys():
                    if old_word != chromesome[old_word]:
                        nb_changed_var += 1
                        nb_changed_pos += len(names_positions_dict.get(old_word, []))

                if self.thld > 0.0:
                    print("**************** best_diff is ok ****************")
                return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, nb_changed_var, nb_changed_pos, chromesome

        # for index, logits in enumerate(mutate_logits):
        #     if mutate_preds[index] != orig_label:
        #         adv_code = map_chromesome(_temp_mutants[index], final_code, self.args.language_type)
        #         for old_word in _temp_mutants[index].keys():
        #             if old_word == _temp_mutants[index][old_word]:
        #                 nb_changed_var += 1
        #                 nb_changed_pos += len(names_positions_dict[old_word])
        #
        #         return code, prog_length, adv_code, true_label, orig_label, mutate_preds[
        #             index], 1, variable_names, None, nb_changed_var, nb_changed_pos, _temp_mutants[index]
        #     _tmp_fitness = max(orig_prob) - logits[orig_label]
        #     mutate_fitness_values.append(_tmp_fitness)
        #
        # # 现在进行替换.
        # for index, fitness_value in enumerate(mutate_fitness_values):
        #     min_value = min(fitness_values)
        #     if fitness_value > min_value:
        #         # 替换.
        #         min_index = fitness_values.index(min_value)
        #         population[min_index] = _temp_mutants[index]
        #         fitness_values[min_index] = fitness_value

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, nb_changed_var, nb_changed_pos, None

    def Alert_GR_thld(self, rank, example, code, label, code_2):

        logits, preds = self.target_model.get_results([example], self.batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        adv_code = ''
        temp_label = None

        # extract identifiers
        identifiers, orig_code_tokens = get_identifiers(code, self.args.language_type)
        variable_names = [iden[0] for iden in identifiers]
        # variable_names = self.filter_identifier(code, identifiers)

        if not orig_label == label:
            print("skip for wrong predict")
            is_success = -4
            return code, len(
                orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        importance_score, replace_token_positions, names_positions_dict = self.get_importance_score(
            code, label, variable_names, code_2)

        if importance_score is None:
            return code, len(
                orig_code_tokens), adv_code, label, orig_label, temp_label, -3, variable_names, None, None, None, None

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

        original_tokens_str = " ".join(orig_code_tokens) if isinstance(orig_code_tokens, list) else orig_code_tokens

        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict.get(tgt_word, [])

            candidates = self.Alert_subs_get(rank, tgt_word)

            if len(candidates) == 0:
                replaced_words[tgt_word] = tgt_word
                continue

            most_gap = 0.0
            candidate = None
            replace_examples = []
            substitute_list = []

            # 存储攻击成功的候选样本信息
            successful_candidates = []  # 格式: (idx, substitute, temp_replace, prob, pred)
            # 存储所有候选样本的文本，用于批量计算相似度
            candidate_texts = []

            for substitute in candidates:
                substitute_list.append(substitute)
                temp_replace = get_example(final_code, tgt_word, substitute, self.args.language_type)
                temp_replace = " ".join(temp_replace.split())

                candidate_texts.append(temp_replace)

                new_feature = load.convert_example_to_features(self.args, temp_replace, label, self.tokenizer,
                                                               self.code_max_len, self.front_token, self.behind_token,
                                                               code_2)
                replace_examples.append(new_feature)

            if len(replace_examples) == 0:
                # 并没有生成新的mutants，直接跳去下一个token
                continue

            new_dataset = self.Dataset(replace_examples)
            # 3. 将他们转化成features
            logits_subs, preds_subs = self.target_model.get_results(new_dataset, self.batch_size)
            assert (len(logits_subs) == len(substitute_list))

            for idx_sub, temp_prob in enumerate(logits_subs):
                temp_label = preds_subs[idx_sub]
                # if temp_label != orig_label:
                if self.is_success(temp_label, orig_label):
                    # 攻击成功，记录候选样本信息
                    successful_candidates.append({
                        'idx': idx_sub,
                        'substitute': substitute_list[idx_sub],
                        'temp_replace': candidate_texts[idx_sub],
                        'prob': temp_prob,
                        'pred': temp_label
                    })

            # 如果有攻击成功的样本，计算相似度并检查阈值
            if successful_candidates:
                # 准备候选代码文本用于相似度计算
                candidate_texts_list = [cand['temp_replace'] for cand in successful_candidates]

                # 批量计算相似度
                similarities, threshold_flags = self.batch_compute_similarities(
                    original_tokens_str,
                    candidate_texts_list
                )

                # 首先检查是否有满足阈值条件的攻击样本
                threshold_success = False
                for i, cand in enumerate(successful_candidates):
                    if threshold_flags[i]:
                        # 满足阈值条件，选择这个攻击样本
                        is_success = 1
                        nb_changed_var += 1
                        nb_changed_pos += len(tgt_positions)
                        candidate = cand['substitute']
                        replaced_words[tgt_word] = candidate
                        adv_code = get_example(final_code, tgt_word, candidate, self.args.language_type)
                        if self.thld > 0.0:
                            print("**************** threshold is ok ****************")
                        print("%s SUC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, candidate,
                               current_prob,
                               cand['prob'][orig_label]), flush=True)
                        threshold_success = True
                        break

                if threshold_success:
                    return code, len(
                        orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

                # 如果没有满足阈值条件的攻击样本，则选择与平均相似度差异最大的攻击样本
                best_diff = -1
                best_candidate = None

                for i, cand in enumerate(successful_candidates):
                    current_diff = abs(similarities[i] - self.avg_sim)
                    if current_diff > best_diff:
                        best_diff = current_diff
                        best_candidate = cand

                if best_candidate is not None:
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(tgt_positions)
                    candidate = best_candidate['substitute']
                    replaced_words[tgt_word] = candidate
                    adv_code = get_example(final_code, tgt_word, candidate, self.args.language_type)

                    print("**************** best_diff is ok ****************")
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_prob,
                           best_candidate['prob'][orig_label]), flush=True)
                    return code, len(
                        orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

            # 如果没有攻击成功的样本，则按照原来的逻辑选择最有可能的替代
            for idx_sub, temp_prob in enumerate(logits_subs):
                temp_label = preds_subs[idx_sub]
                # 没有标签变化: 测量概率差
                gap = current_prob - temp_prob[temp_label]
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute_list[idx_sub]

                #     # attack success
                #     is_success = 1
                #     nb_changed_var += 1
                #     nb_changed_pos += len(tgt_positions)
                #     candidate = substitute_list[idx_sub]
                #     replaced_words[tgt_word] = candidate
                #     adv_code = get_example(final_code, tgt_word, candidate, self.args.language_type)
                #
                #     print("%s SUC! %s => %s (%.5f => %.5f)" % \
                #           ('>>', tgt_word, candidate,
                #            current_prob,
                #            temp_prob[orig_label]), flush=True)
                #     return code, len(
                #         orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                # else:
                #     # no label change: measure prob gap
                #     gap = current_prob - temp_prob[temp_label]
                #     if gap > most_gap:
                #         most_gap = gap
                #         candidate = substitute_list[idx_sub]

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
            orig_code_tokens), adv_code, label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words


