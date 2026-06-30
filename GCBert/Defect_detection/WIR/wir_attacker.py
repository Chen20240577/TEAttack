import sys

sys.path.append('../../../')
sys.path.append('../../../python_parser')
sys.path.append('../saved_models/')

import copy
import random
from run import TextDataset, InputFeatures, extract_dataflow
from utils import is_valid_variable_name, _tokenize, \
    get_identifier_posistions_from_code, get_masked_code_by_position

from utils import GraphCodeDataset, get_code_tokens
from utils import isUID
from run_parser import get_identifiers, get_example


# convert_examples_to_features(new_code, example[3].item(), tokenizer, args)
def convert_examples_to_features(code, label, tokenizer, args):
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


def get_importance_score(args, example, code, words_list: list, variable_names: list,
                         tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.

    for index, code_tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(code_tokens)
        new_feature = convert_examples_to_features(new_code, example[3].item(), tokenizer, args)
        new_example.append(new_feature)
    new_dataset = GraphCodeDataset(new_example, args)
    # 3. 将他们转化成features
    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]
    # 第一个是original code的数据.

    orig_prob = max(orig_probs)
    # predicted label对应的probability

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions


class WIR_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, _token2idx, _idx2token) -> None:
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args

    def filter_identifier(self, code, identifiers):
        code_token = get_code_tokens(code, self.tokenizer_tgt, self.args.language_type)
        filter_identifiers = []
        for identifier in identifiers:
            if is_valid_variable_name(identifier, self.args.language_type):
                position = []
                for index, token in enumerate(code_token):
                    if identifier == token:
                        position.append(index)
                if not all(x > self.args.data_flow_length + self.args.code_length - 2 for x in position):
                    filter_identifiers.append(identifier)
        return filter_identifiers

    def wir_random_attack(self, example, code, rank, recoder):

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[3].item()
        adv_code = ''
        temp_label = None

        # When do attack, we only attack the first code snippet
        identifiers, code_tokens = get_identifiers(code, self.args.language_type)  # 只得到code_1中的identifier
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_tgt)

        variable_names = [identifier[0] for identifier in identifiers]
        variable_names = self.filter_identifier(code, variable_names)

        if not orig_label == true_label:
            print("skip for wrong predict")
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args, example,
                                                                                               processed_code,
                                                                                               words,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               [0, 1],
                                                                                               batch_size=self.args.eval_batch_size,
                                                                                               max_length=self.args.data_flow_length + self.args.code_length,
                                                                                               model_type='classification')

        if importance_score is None:
            return code, prog_length, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None

        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        # 根据importance_score进行排序

        final_code = copy.deepcopy(code)

        nb_changed_var = 0  # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict[tgt_word]  # the positions of tgt_word in code

            all_substitues = []
            num = 0
            while num < 30:
                tmp_var = random.choice(self.idx2token)
                if isUID(tmp_var):
                    all_substitues.append(tmp_var)
                    num += 1

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            # 依次记录了被加进来的substitue
            # 即，每个temp_replace对应的substitue.
            for substitute in all_substitues:
                substitute_list.append(substitute)
                # 记录了替换的顺序
                # temp_replace = get_example(final_code, tgt_word, substitute)
                # 需要添加一个语言类型参数 'python'
                temp_replace = get_example(final_code, tgt_word, substitute, self.args.language_type)

                # recoder.write(rank, code, temp_replace, true_label)

                temp_replace = " ".join(temp_replace.split())
                # temp_replace = self.tokenizer_tgt.tokenize(temp_replace)
                # 需要将几个位置都替换成sustitue_
                new_feature = convert_examples_to_features(temp_replace,
                                                           example[3].item(),
                                                           self.tokenizer_tgt,
                                                           self.args)
                replace_examples.append(new_feature)
            if len(replace_examples) == 0:
                # 并没有生成新的mutants，直接跳去下一个token
                continue

            new_dataset = GraphCodeDataset(replace_examples, self.args)
            # 3. 将他们转化成features
            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert (len(logits) == len(substitute_list))

            for index, temp_prob in enumerate(logits):
                temp_label = preds[index]
                if temp_label != orig_label:
                    # 如果label改变了，说明这个mutant攻击成功
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate

                    # adv_code = get_example(final_code, tgt_word, candidate)
                    # 需要添加一个语言类型参数 'python'
                    adv_code = get_example(final_code, tgt_word, candidate, self.args.language_type)

                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_prob,
                           temp_prob[orig_label]), flush=True)
                    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]

            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                # final_code = get_example(final_code, tgt_word, candidate)
                # 需要添加一个语言类型参数 'python'
                final_code = get_example(final_code, tgt_word, candidate, self.args.language_type)
                replaced_words[tgt_word] = candidate
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                      ('>>', tgt_word, candidate,
                       current_prob + most_gap,
                       current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

            adv_code = final_code

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
