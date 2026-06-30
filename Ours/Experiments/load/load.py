import pickle

import joblib
import torch
import json
import os
import sys
import re

sys.path.append("../../../")
sys.path.append('../../../python_parser')
# 因为python_parser放在对应的目录不好修改
# 修改会导致本实验中其他攻击方法在对应目录下不能运行
# 所以，这个辅助脚本也放在了同层目录下

from CodeBert.Authorship_Attribution.saved_models.model import Model as bert_authorship_model
from CodeBert.Clone_detection.saved_models.model import Model as bert_clone_model
from CodeBert.Defect_detection.saved_models.model import Model as bert_defect_model

from GCBert.Authorship_Attribution.saved_models.model import Model as gcbert_authorship_model
from GCBert.Clone_detection.saved_models.model import Model as gcbert_clone_model
from GCBert.Defect_detection.saved_models.model import Model as gcbert_defect_model

from CodeGPT.Authorship_Attribution.saved_models.model import Model as gpt2_authorship_model
from CodeGPT.Clone_detection.saved_models.model import Model as gpt2_clone_model
from CodeGPT.Defect_detection.saved_models.model import Model as gpt2_defect_model

from CodeT5.Authorship_Attribution.saved_models.model import Model as t5_authorship_model
from CodeT5.Clone_detection.saved_models.model import Model as t5_clone_model
from CodeT5.Defect_detection.saved_models.model import Model as t5_defect_model


from CodeBert.Authorship_Attribution.saved_models.run import TextDataset as bert_authorship_TextDataset
from CodeBert.Clone_detection.saved_models.run import TextDataset as bert_clone_TextDataset
from CodeBert.Defect_detection.saved_models.run import TextDataset as bert_defect_TextDataset

from GCBert.Authorship_Attribution.saved_models.run import TextDataset as gcbert_authorship_TextDataset
from GCBert.Clone_detection.saved_models.run import TextDataset as gcbert_clone_TextDataset
from GCBert.Defect_detection.saved_models.run import TextDataset as gcbert_defect_TextDataset

from CodeGPT.Authorship_Attribution.saved_models.run import TextDataset as gpt2_authorship_TextDataset
from CodeGPT.Clone_detection.saved_models.run import TextDataset as gpt2_clone_TextDataset
from CodeGPT.Defect_detection.saved_models.run import TextDataset as gpt2_defect_TextDataset

from CodeT5.Authorship_Attribution.saved_models.run import TextDataset as t5_authorship_TextDataset
from CodeT5.Clone_detection.saved_models.run import TextDataset as t5_clone_TextDataset
from CodeT5.Defect_detection.saved_models.run import TextDataset as t5_defect_TextDataset


from CodeBert.Authorship_Attribution.saved_models.run import InputFeatures as bert_authorship_InputFeatures
from CodeBert.Clone_detection.saved_models.run import InputFeatures as bert_clone_InputFeatures
from CodeBert.Defect_detection.saved_models.run import InputFeatures as bert_defect_InputFeatures

from GCBert.Authorship_Attribution.saved_models.run import InputFeatures as gcbert_authorship_InputFeatures
from GCBert.Clone_detection.saved_models.run import InputFeatures as gcbert_clone_InputFeatures
from GCBert.Defect_detection.saved_models.run import InputFeatures as gcbert_defect_InputFeatures

from CodeGPT.Authorship_Attribution.saved_models.run import InputFeatures as gpt2_authorship_InputFeatures
from CodeGPT.Clone_detection.saved_models.run import InputFeatures as gpt2_clone_InputFeatures
from CodeGPT.Defect_detection.saved_models.run import InputFeatures as gpt2_defect_InputFeatures

from CodeT5.Authorship_Attribution.saved_models.run import InputFeatures as t5_authorship_InputFeatures
from CodeT5.Clone_detection.saved_models.run import InputFeatures as t5_clone_InputFeatures
from CodeT5.Defect_detection.saved_models.run import InputFeatures as t5_defect_InputFeatures



from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_c import DFG_c
from parser_folder.DFG_java import DFG_java
from parser_folder.DFG import DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser_folder import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from run_parser import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript,
    'c':DFG_c,
}

#load parsers
parsers={}
for lang in dfg_function:
    # LANGUAGE = Language('parser/my-languages.so', lang)
    LANGUAGE = Language('../../../python_parser/parser_folder/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser,dfg_function[lang]]
    parsers[lang]= parser

def extract_dataflow(code, parser, lang):
    #remove comments
    code_tokens=""
    code = code.replace("\\n", "\n")
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"
    try:
        tree = parser[0].parse(bytes(code,'utf8'))
        root_node = tree.root_node
        tokens_index=tree_to_token_index(root_node)
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)
        try:
            DFG,_=parser[1](root_node,index_to_code,{})
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

import transformers
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,RobertaForSequenceClassification,
                          T5Config, T5ForConditionalGeneration,
                          AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer)

MODEL_CLASSES = {
    'bert': (RobertaConfig, RobertaModel, RobertaTokenizer, "../../../../../Models/FacebookAI/roberta-base"),
    'gcbert': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, "../../../../../Models/microsoft/graphcodebert-base"),
    't5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer, "../../../../../Models/Salesforce/codet5-base"),
    'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer, "../../../../../Models/microsoft/CodeGPT-small-java-adaptedGPT2"),
}

# task, model_type, output_dir, checkpoint_prefix, device
def load_model(args, output_dir):
    config_class, model_class, tokenizer_class, model_path = MODEL_CLASSES[args.model_type]

    if args.model_type == 'bert' and args.task =='defect':
        model_class = RobertaForSequenceClassification

    config = config_class.from_pretrained(model_path, cache_dir=None)

    config.num_labels = args.number_labels

    tokenizer = tokenizer_class.from_pretrained(model_path,
                                                do_lower_case=False,
                                                cache_dir=None)
    if args.model_type == 'gcbert':
        # gcbert用的是 data_flow_length和code_length
        pass
    else:
        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if args.model_type == 'gpt2':
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # 用结束符作为填充符
        config.pad_token_id = tokenizer.pad_token_id

    model = model_class.from_pretrained(model_path,
                                        from_tf=bool('.ckpt' in model_path),
                                        config=config,
                                        cache_dir=None)

    target_model = eval(f"{args.model_type}_{args.task}_model")(model, config, tokenizer, args)
    target_model_path = os.path.join(output_dir, '{}'.format(args.checkpoint_prefix))
    target_model.load_state_dict(torch.load(target_model_path))
    target_model.to(args.device)

    return target_model, tokenizer, args


def load_tfidf(task):
    task_name = task_name_get(task)
    model_path = ("../../../TF_IDF/" + task_name + "/tfidf.pkl")
    vectorizer = joblib.load(model_path)
    return vectorizer


def load_code(args):
    source_codes = []
    substs = []

    if args.task == 'authorship':
        file_type = args.data_file.split('/')[-1].split('.')[0]  # valid
        folder = '/'.join(args.data_file.split('/')[:-1])  # 得到文件目录
        codes_file_path = os.path.join(folder, '{}_subs.jsonl'.format(
            file_type))
        print(codes_file_path)
        with open(codes_file_path) as rf:
            for line in rf:
                item = json.loads(line.strip())
                source_codes.append(item["code"].replace("\\n", "\n").replace('\"', '"'))
                substs.append(item["substitutes"])

    elif args.task == 'clone':
        ## Load code pairs
        postfix = args.data_file.split('/')[-1].split('.txt')[0]
        folder = '/'.join(args.data_file.split('/')[:-1])  # 得到文件目录
        code_pairs_file_path = os.path.join(folder, 'cached_{}.pkl'.format(
            postfix))
        with open(code_pairs_file_path, 'rb') as f:
            source_codes = pickle.load(f)

        postfix = postfix.split("_")
        subs_path = os.path.join(folder, 'test_subs_{}_{}.jsonl'.format(
            postfix[-2], postfix[-1]))
        with open(subs_path) as f:
            for line in f:
                js = json.loads(line.strip())
                substs.append(js["substitutes"])

    elif args.task == 'defect':
        with open(args.data_file) as f:
            for line in f:
                js = json.loads(line.strip())
                # code = ' '.join(js['func'].split())
                code = js['func']
                source_codes.append(code)
                substs.append(js['substitutes'])

    return source_codes, substs


def load_dataset(args, tokenizer):
    dataset = eval(f"{args.model_type}_{args.task}_TextDataset")(tokenizer, args, args.data_file)
    return dataset


def convert_example_to_features(args, code, label, tokenizer, code_max_len, front_token, behind_token, code_2):
        """
        将代码字符串转成 InputFeatures（与 run.py 的输入格式保持兼容）
        """
        # tokenize then clip to block

        # if self.task == 'authorship' or self.task == 'clone':
        InputFeatures = eval(f"{args.model_type}_{args.task}_InputFeatures")

        feature = None
        source_tokens, source_ids = None, None
        position_idx, dfg_to_code, dfg_to_dfg = None, None, None

        if args.task == 'authorship' or args.task == 'defect':
            if args.model_type == 'bert' or args.model_type == 't5' or args.model_type == 'gpt2':
                source_tokens, source_ids = convert_code_to_features(code, tokenizer, code_max_len, front_token, behind_token)
            elif args.model_type == 'gcbert':
                source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg = gcbert_convert_code_to_features(code, tokenizer, code_max_len, front_token, behind_token, args)

            if args.model_type == 'bert' or args.model_type == 't5':
                feature = InputFeatures(source_tokens, source_ids, 0, label)
            elif args.model_type == 'gcbert':
                if args.task == 'authorship':
                    feature = InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, label)
                elif args.task == 'defect':
                    feature = InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, 0, label)
            elif args.model_type == 'gpt2':
                attention_mask = (source_ids != 0)
                feature = InputFeatures(source_tokens, source_ids, attention_mask, 0, label)


        elif args.task == 'clone':
            source_tokens_2, source_ids_2 = None, None
            position_idx_2, dfg_to_code_2, dfg_to_dfg_2 = None, None, None

            if args.model_type == 'bert' or args.model_type == 't5' or args.model_type == 'gpt2':
                source_tokens, source_ids = convert_code_to_features(code, tokenizer, code_max_len, front_token, behind_token)
                source_tokens_2, source_ids_2 = convert_code_to_features(code_2, tokenizer, code_max_len, front_token, behind_token)
                source_tokens = source_tokens + source_tokens_2
                source_ids = source_ids + source_ids_2
            elif args.model_type == 'gcbert':
                source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg \
                    = gcbert_convert_code_to_features(code, tokenizer, code_max_len, front_token, behind_token, args)
                source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2 \
                    = gcbert_convert_code_to_features(code_2, tokenizer, code_max_len, front_token, behind_token, args)

            if args.model_type == 'bert' or args.model_type == 't5':
                feature = InputFeatures(source_tokens, source_ids, label, 0, 0)
            elif args.model_type == 'gcbert':
                feature = InputFeatures(source_tokens,source_ids,
                                        position_idx,dfg_to_code,dfg_to_dfg,
                                        source_tokens_2,source_ids_2,
                                        position_idx_2,dfg_to_code_2,dfg_to_dfg_2,
                                        label, 0, 0)
            elif args.model_type == 'gpt2':
                attention_mask = (source_ids != 0)
                feature = InputFeatures(source_tokens, source_ids, attention_mask, label, 0, 0)

        return feature


def convert_code_to_features(code, tokenizer, code_max_len, front_token, behind_token):
    code_tokens = tokenizer.tokenize(code)[:code_max_len - 2]
    source_tokens = [front_token] + code_tokens + [behind_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = code_max_len - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_tokens, source_ids


def gcbert_convert_code_to_features(code, tokenizer, code_max_len, front_token, behind_token, args):
    parser = parsers[args.language_type]
    code_tokens, dfg = extract_dataflow(code, parser, args.language_type)

    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    code_tokens = code_tokens[:code_max_len - 2 - min(len(dfg), args.data_flow_length)]
    source_tokens = [front_token] + code_tokens + [behind_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:code_max_len - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = code_max_len - len(source_ids)
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

    return source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg


from copy import deepcopy
def conduct_example(args, code, label, tokenizer, code_max_len, front_token, behind_token, adv_code=None):
    code_1 = None
    code_2 = None

    if args.task == 'clone':
        code_1 = deepcopy(code[2])
        code_2 = deepcopy(code[3])
        if adv_code is not None:
            code_2 = adv_code
    elif args.task == 'authorship' or args.task == 'defect':
        code_1 = code
        if adv_code is not None:
            code_1 = adv_code

    # words, pre_words = pre_code(code_1)
    # pre_words_2 = None

    # if code_2 is not None:
    #     words_2, pre_words_2 = pre_code(code_2)

    code_1 = code_1.replace("\\n", "\n").replace('\"', '"')
    if code_2 is not None:
        code_2 = code_2.replace("\\n", "\n").replace('\"', '"')
    # return convert_example_to_features(args, pre_words, label, tokenizer, code_max_len, front_token, behind_token, pre_words_2)
    return convert_example_to_features(args, code_1, label, tokenizer, code_max_len, front_token, behind_token, code_2)

def model_name_get(model_type):
    if model_type == "bert":
        model_name = "CodeBert"
    elif model_type == "gcbert":
        model_name = "GCBert"
    elif model_type == "t5":
        model_name = "CodeT5"
    elif model_type == "gpt2":
        model_name = "CodeGPT"
    else:
        model_name = None
    return model_name


def task_name_get(task):
    if task == "authorship":
        task_name = "Authorship_Attribution"
    elif task == "clone":
        task_name = "Clone_detection"
    elif task == "defect":
        task_name = "Defect_detection"
    else:
        task_name = None
    return task_name


def label_get(example, model_type, task):
    if model_type == "gcbert":
        if task == "authorship" or task == "defect":
            true_label = example[3].item()
            pass
        elif task == "clone":
            true_label = example[6].item()
    elif model_type == "bert" or model_type == "t5" or model_type == "gpt2":
        true_label = example[1].item()
    else:
        true_label = None
    return true_label


def pre_code(code):
    if isinstance(code, str):
        words = code.split()
    else:
        words = code
    pre_words = " ".join(words)
    return words, pre_words


def date_file_get(task, model):
    task_name = task_name_get(task)

    if task == "authorship":
        date_file = "../../../Datasets/" + task_name + dataset_name[model] + "/processed_gcjpy/valid.txt"
    elif task == "clone":
        date_file = "../../../Datasets/" + task_name + dataset_name[model] + "/data_folder/test_sampled_0_500.txt"
    elif task == "defect":
        date_file = "../../../Datasets/" + task_name + dataset_name[model] + "/dataset/test_subs_0_400.jsonl"

    return date_file


def batch_size_get(task, model):

    batch_size = batch_sizes[model]

    if task == "authorship" or task == "defect":
        batch_size = batch_size * 2
    elif task == "clone":
        batch_size = batch_size

    return batch_size


def nearest_k_path_get(task, K, attack_type):
    task_name = task_name_get(task)
    if "wordnet" in attack_type:
        nearest_k_path = ("../../../WN_Models/" + task_name +
                          "/datasets/var_name/code_nearest_top" + str(K) + "_wordnet_compatible.pkl")
    else:
        nearest_k_path = ("../../../W2V_Models/" + task_name +
                          "/datasets/var_name/code_nearest_top" + str(K) + ".pkl")
    return nearest_k_path

def word2vec_path_get(task):
    task_name = task_name_get(task)
    word2vec_path = ("../../../W2V_Models/" + task_name +
                      "/word2vec_model/node_w2v_code_64.model")
    return word2vec_path


dataset_name = {
    'bert'  :'/codebert-mlm',
    'gcbert':'/graphcodebert-base',
    't5'    :'/codet5-base',
    'gpt2'  :'/codegpt-small',
}

batch_sizes = {
    'bert'   :16,
    'gcbert' :16,
    't5'     :4,
    'gpt2'   :8,
}

languages = {
    'authorship'    :'python',
    'clone'         :'java',
    'defect'        :'c',
}

number_labels = {
    'authorship'   :66,
    'clone'        :2,
    'defect'       :1,
}

store_name = {
    "bert"    :"Bert",
    "gcbert"  :"GCBert",
    "t5"      :"T5",
    "gpt2"    :"GPT",
}


