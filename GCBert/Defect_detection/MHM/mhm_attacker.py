import sys

sys.path.append('../../../')
sys.path.append('../../../python_parser')

import copy
import random
from model import Model
from run import TextDataset, InputFeatures, extract_dataflow
from utils import _tokenize, \
    get_identifier_posistions_from_code

from utils import GraphCodeDataset
from utils import isUID
from run_parser import get_identifiers, get_example


def convert_examples_to_features(code, tokenizer, label, args):
    # source
    dfg, index_table, code_tokens = extract_dataflow(code, "c")

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

    return InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, 0, label)


class MHM_Attacker():
    def __init__(self, args, model_tgt, model_mlm, tokenizer_mlm, _token2idx, _idx2token) -> None:
        self.classifier = model_tgt
        self.model_mlm = model_mlm
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args
        self.tokenizer_mlm = tokenizer_mlm

    def mcmc(self, tokenizer, substituions,
             index, recoder, code=None, _label=None, _n_candi=30,
             _max_iter=100, _prob_threshold=0.95):

        orig_code = code

        identifiers, code_tokens = get_identifiers(code, 'c')
        prog_length = len(code_tokens)
        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, tokenizer)
        raw_tokens = copy.deepcopy(words)
        variable_names = list(substituions.keys())

        uid = get_identifier_posistions_from_code(words, variable_names)

        if len(uid) <= 0:  # 是有可能存在找不到变量名的情况的.
            return {'succ': None, 'tokens': None, 'raw_tokens': None}

        variable_substitue_dict = {}

        for tgt_word in uid.keys():
            variable_substitue_dict[tgt_word] = substituions[tgt_word]

        if len(variable_substitue_dict) <= 0:  # 是有可能存在找不到变量名的情况的.
            return {'succ': None, 'tokens': None, 'raw_tokens': None}

        old_uids = {}
        old_uid = ""
        for iteration in range(1, 1 + _max_iter):
            # 这个函数需要tokens
            res = self.__replaceUID(_tokens=code, _label=_label, _uid=uid,
                                    substitute_dict=variable_substitue_dict,
                                    _n_candi=_n_candi,
                                    _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")

            # if res["new_pred"] == _label:
            #     recoder.write(index, orig_code, res['tokens'], _label)

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
                uid[res['new_uid']] = uid.pop(res['old_uid'])  # 替换key，但保留value.
                variable_substitue_dict[res['new_uid']] = variable_substitue_dict.pop(res['old_uid'])
                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    replace_info = {}
                    nb_changed_pos = 0
                    for uid_ in old_uids.keys():
                        # 这两行是添加的加容错的
                        # **************************************
                        current_uid = old_uids[uid_][-1]
                        if current_uid not in uid:
                            print(f"Warning: {current_uid} not found in uid dictionary. Skipping.")
                            continue
                        replace_info[uid_] = current_uid
                        nb_changed_pos += len(uid[current_uid])
                        # **************************************

                        # replace_info[uid_] = old_uids[uid_][-1]
                        # nb_changed_pos += len(uid[old_uids[uid_][-1]])

                    return {'succ': True, 'tokens': code,
                            'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"],
                            "is_success": 1, "old_uid": old_uid, "score_info": res["old_prob"][0] - res["new_prob"][0],
                            "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos,
                            "replace_info": replace_info, "attack_type": "MHM"}
        replace_info = {}
        nb_changed_pos = 0
        for uid_ in old_uids.keys():
            # 这两行是添加的加容错的
            # **************************************
            current_uid = old_uids[uid_][-1]
            if current_uid not in uid:
                print(f"Warning: {current_uid} not found in uid dictionary. Skipping.")
                continue
            replace_info[uid_] = current_uid
            nb_changed_pos += len(uid[current_uid])
            # **************************************

            # replace_info[uid_] = old_uids[uid_][-1]
            # nb_changed_pos += len(uid[old_uids[uid_][-1]])

        return {'succ': False, 'tokens': res['tokens'], 'raw_tokens': None, "prog_length": prog_length,
                "new_pred": res["new_pred"], "is_success": -1, "old_uid": old_uid,
                "score_info": res["old_prob"][0] - res["new_prob"][0], "nb_changed_var": len(old_uids),
                "nb_changed_pos": nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM"}

    def mcmc_random(self, tokenizer, substituions,
                    index, recoder, code=None, _label=None, _n_candi=30,
                    _max_iter=100, _prob_threshold=0.95):

        orig_code = code

        identifiers, code_tokens = get_identifiers(code, 'c')
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, tokenizer)
        raw_tokens = copy.deepcopy(words)
        variable_names = list(substituions.keys())

        uid = get_identifier_posistions_from_code(words, variable_names)

        if len(uid) <= 0:  # 是有可能存在找不到变量名的情况的.
            return {'succ': None, 'tokens': None, 'raw_tokens': None}

        variable_substitue_dict = {}
        for tgt_word in uid.keys():
            variable_substitue_dict[tgt_word] = substituions[tgt_word]

        old_uids = {}
        old_uid = ""
        for iteration in range(1, 1 + _max_iter):
            # 这个函数需要tokens
            res = self.__replaceUID_random(_tokens=code, _label=_label, _uid=uid,
                                           substitute_dict=variable_substitue_dict,
                                           _n_candi=_n_candi,
                                           _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")

            # if res["new_pred"] == _label:
            #     recoder.write(index, orig_code, res['tokens'], _label)

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
                uid[res['new_uid']] = uid.pop(res['old_uid'])  # 替换key，但保留value.
                variable_substitue_dict[res['new_uid']] = variable_substitue_dict.pop(res['old_uid'])

                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    replace_info = {}
                    nb_changed_pos = 0
                    for uid_ in old_uids.keys():
                        # 这两行是添加的加容错的
                        # **************************************
                        current_uid = old_uids[uid_][-1]
                        if current_uid not in uid:
                            print(f"Warning: {current_uid} not found in uid dictionary. Skipping.")
                            continue
                        replace_info[uid_] = current_uid
                        nb_changed_pos += len(uid[current_uid])
                        # **************************************

                        # replace_info[uid_] = old_uids[uid_][-1]
                        # nb_changed_pos += len(uid[old_uids[uid_][-1]])

                    return {'succ': True, 'tokens': code,
                            'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"],
                            "is_success": 1, "old_uid": old_uid, "score_info": res["old_prob"][0] - res["new_prob"][0],
                            "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos,
                            "replace_info": replace_info, "attack_type": "MHM-Origin"}
        replace_info = {}
        nb_changed_pos = 0

        for uid_ in old_uids.keys():
            # 这两行是添加的加容错的
            # **************************************
            current_uid = old_uids[uid_][-1]
            if current_uid not in uid:
                print(f"Warning: {current_uid} not found in uid dictionary. Skipping.")
                continue
            replace_info[uid_] = current_uid
            nb_changed_pos += len(uid[current_uid])
            # **************************************

            # replace_info[uid_] = old_uids[uid_][-1]
            # nb_changed_pos += len(uid[old_uids[uid_][-1]])

        return {'succ': False, 'tokens': res['tokens'], 'raw_tokens': None, "prog_length": prog_length,
                "new_pred": res["new_pred"], "is_success": -1, "old_uid": old_uid,
                "score_info": res["old_prob"][0] - res["new_prob"][0], "nb_changed_var": len(old_uids),
                "nb_changed_pos": nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM-Origin"}

    def __replaceUID(self, _tokens, _label=None, _uid={}, substitute_dict={},
                     _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):

        assert _candi_mode.lower() in ["random", "nearby"]

        selected_uid = random.sample(substitute_dict.keys(), 1)[0]  # 选择需要被替换的变量名
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(substitute_dict[selected_uid],
                                   min(_n_candi, len(substitute_dict[selected_uid]))):  # 选出_n_candi数量的候选.
                if c in _uid.keys():
                    continue
                if isUID(c):  # 判断是否是变量名.
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c, "c")
                    # for i in _uid[selected_uid]: # 依次进行替换.
                    #     if i >= len(candi_tokens[-1]):
                    #         break
                    #     candi_tokens[-1][i] = c # 替换为新的candidate.
            new_example = []
            for tmp_tokens in candi_tokens:
                tmp_code = tmp_tokens
                new_feature = convert_examples_to_features(tmp_code, self.tokenizer_mlm, _label, self.args)
                new_example.append(new_feature)
            new_dataset = GraphCodeDataset(new_example, self.args)
            prob, pred = self.classifier.get_results(new_dataset, self.args.eval_batch_size)

            for i in range(len(candi_token)):  # Find a valid example
                if pred[i] != _label:  # 如果有样本攻击成功
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": prob[0], "new_prob": prob[i],
                            "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}

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
        else:
            pass

    def __replaceUID_random(self, _tokens, _label=None, _uid={}, substitute_dict={},
                            _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):

        assert _candi_mode.lower() in ["random", "nearby"]

        selected_uid = random.sample(substitute_dict.keys(), 1)[0]  # 选择需要被替换的变量名
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(self.idx2token, _n_candi):  # 选出_n_candi数量的候选.
                if c in _uid.keys():
                    continue
                if isUID(c):  # 判断是否是变量名.
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c, "c")
                    # for i in _uid[selected_uid]: # 依次进行替换.
                    #     if i >= len(candi_tokens[-1]):
                    #         break
                    #     candi_tokens[-1][i] = c # 替换为新的candidate.

            new_example = []
            for tmp_tokens in candi_tokens:
                tmp_code = tmp_tokens
                new_feature = convert_examples_to_features(tmp_code, self.tokenizer_mlm, _label, self.args)
                new_example.append(new_feature)
            new_dataset = GraphCodeDataset(new_example, self.args)
            prob, pred = self.classifier.get_results(new_dataset, self.args.eval_batch_size)

            for i in range(len(candi_token)):  # Find a valid example
                if pred[i] != _label:  # 如果有样本攻击成功
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": prob[0], "new_prob": prob[i],
                            "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}

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
        else:
            pass

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
