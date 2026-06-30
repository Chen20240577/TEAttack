"""
基于WordNet的变量相似度计算和最近邻查找
替换原有的Word2Vec模型，使用WordNet语义关系
优化版本：解决内存爆炸问题
修改版本：与Word2Vec版本保存格式完全一致
"""

import gc
import os
import re
import sys
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn

# 添加项目路径
sys.path.append('../../../')
sys.path.append('../../../python_parser')

# 下载WordNet数据（首次运行需要）
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

# 配置参数
k = 30  # 最近邻数量
root = '../'


class WordNetSimilarity:
    """WordNet相似度计算类（内存优化版本）"""

    def __init__(self, max_cache_size=10000, max_syn_cache=5000):
        # 预编译正则表达式提高效率
        self._word_split_regex = re.compile('[A-Z][a-z]*|[a-z]+')
        self._non_alpha_regex = re.compile(r'[^a-zA-Z]')

        # 使用lru_cache自动管理缓存大小
        self.semantic_similarity_cached = lru_cache(maxsize=max_cache_size)(self._semantic_similarity_uncached)
        self._get_synonyms_cached = lru_cache(maxsize=max_syn_cache)(self._get_synonyms_uncached)

    def preprocess_identifier(self, identifier):
        """
        预处理代码标识符以适应WordNet
        使用预编译正则表达式提高性能
        """
        if not identifier or not isinstance(identifier, str):
            return ""

        # 移除数字和特殊字符
        identifier = self._non_alpha_regex.sub(' ', identifier)

        # 拆分驼峰命名法和大写字母
        words = self._word_split_regex.findall(identifier)

        if words:
            # 使用最后一个单词作为主要关键词（通常是最有意义的）
            main_word = words[-1].lower()
            return main_word
        return identifier.lower()

    def _get_synonyms_uncached(self, word):
        """获取单词的同义词集（无缓存版本）"""
        synonyms = set()
        try:
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().lower().replace('_', ' ')
                    if synonym != word and len(synonym.split()) == 1:  # 只保留单词同义词
                        synonyms.add(synonym)
        except Exception:
            pass
        return list(synonyms)

    def get_synonyms(self, word):
        """获取单词的同义词集（带缓存）"""
        return self._get_synonyms_cached(word)

    def _semantic_similarity_uncached(self, word1, word2):
        """
        计算两个单词的语义相似度核心逻辑（无缓存版本）
        使用路径相似度算法
        """
        if word1 == word2:
            return 1.0

        # 预处理单词
        processed1 = self.preprocess_identifier(word1)
        processed2 = self.preprocess_identifier(word2)

        # 如果预处理后相同
        if processed1 == processed2:
            return 1.0

        try:
            synsets1 = wn.synsets(processed1)
            synsets2 = wn.synsets(processed2)

            if not synsets1 or not synsets2:
                # 尝试通过同义词扩展
                synonyms1 = self.get_synonyms(processed1)
                synonyms2 = self.get_synonyms(processed2)

                # 检查同义词交集
                common_synonyms = set(synonyms1) & set(synonyms2)
                if common_synonyms:
                    return 0.6
                else:
                    return 0.0

            max_similarity = 0.0
            # 使用生成器表达式减少内存使用
            similarities = (syn1.path_similarity(syn2)
                            for syn1 in synsets1
                            for syn2 in synsets2)

            for similarity in similarities:
                if similarity and similarity > max_similarity:
                    max_similarity = similarity
                    if max_similarity > 0.8:  # 提前终止，如果找到高相似度
                        break

            return max_similarity

        except Exception as e:
            print(f"计算相似度时出错: {e}, 单词1: {word1}, 单词2: {word2}")
            return 0.0

    def semantic_similarity(self, word1, word2):
        """语义相似度计算（带缓存）"""
        return self.semantic_similarity_cached(word1, word2)


def get_topk_index(k, scores):
    """
    获取top-k索引，优化内存使用
    """
    if len(scores) == 0:
        return []

    # 使用argpartition而不是argsort来提高性能，特别是当k远小于n时
    if k < len(scores) // 10:  # 当k相对较小时使用argpartition
        scores_array = np.array(scores)
        # 获取最大的k个值的索引
        top_k_indices = np.argpartition(scores_array, -k)[-k:]
        # 对这k个索引按分数降序排序
        top_k_indices = top_k_indices[np.argsort(scores_array[top_k_indices])[::-1]]
        return top_k_indices
    else:
        # 当k较大时使用传统的argsort
        scores_array = np.array(scores)
        return scores_array.argsort()[::-1][:k]


def find_wordnet_nearest_variables(target_vars, all_vars, k=30, formal_params=None, exclude_vars=None):
    """
    使用WordNet寻找最近邻变量（与Word2Vec版本格式一致）

    参数:
    target_vars: 目标变量列表
    all_vars: 所有变量列表
    k: 最近邻数量
    formal_params: 形式参数列表（用于特殊处理）
    exclude_vars: 需要排除的变量列表（当前代码中的变量）
    """
    wn_sim = WordNetSimilarity()
    nearest_dict = {}

    # 使用集合进行快速查找
    formal_params_set = set(formal_params) if formal_params else set()
    exclude_vars_set = set(exclude_vars) if exclude_vars else set()
    all_vars_set = set(all_vars)  # 用于快速查找

    # 预先计算排除索引
    exclude_indices = set()
    for i, var in enumerate(all_vars):
        if var in formal_params_set or var in exclude_vars_set:
            exclude_indices.add(i)

    print(f"开始处理 {len(target_vars)} 个目标变量...")

    for i, target_var in enumerate(target_vars):
        if target_var not in all_vars_set:
            print(f"变量 '{target_var}' 不在所有变量列表中，跳过")
            nearest_dict[target_var] = []
            continue

        if i % 100 == 0:  # 每100个变量清理一次内存
            gc.collect()
            print(f"已处理 {i}/{len(target_vars)} 个变量")

        target_index = all_vars.index(target_var)

        # 使用生成器表达式计算相似度，避免创建大型临时列表
        similarity_scores = []
        valid_candidates = []

        for j, candidate_var in enumerate(all_vars):
            if j == target_index or j in exclude_indices:
                continue

            similarity = wn_sim.semantic_similarity(target_var, candidate_var)
            if similarity > 0:  # 只保留有相似度的候选
                similarity_scores.append(similarity)
                valid_candidates.append(candidate_var)

        # 获取top-k最近邻
        if len(similarity_scores) == 0:
            nearest_dict[target_var] = []
            # 尝试同义词扩展
            synonyms = wn_sim.get_synonyms(wn_sim.preprocess_identifier(target_var))
            nearest_vars = []
            for synonym in synonyms:
                if (synonym in all_vars_set and synonym != target_var
                        and synonym not in exclude_vars_set and synonym not in formal_params_set):
                    nearest_vars.append(synonym)
                    if len(nearest_vars) >= k:
                        break
            nearest_dict[target_var] = nearest_vars[:k]
            continue

        # 使用优化后的top-k索引查找
        if len(similarity_scores) <= k:
            top_k_indices = list(range(len(similarity_scores)))
        else:
            top_k_indices = get_topk_index(k, similarity_scores)

        nearest_vars = [valid_candidates[idx] for idx in top_k_indices]

        # 如果找到的最近邻不足k个，尝试同义词扩展
        if len(nearest_vars) < k:
            synonyms = wn_sim.get_synonyms(wn_sim.preprocess_identifier(target_var))
            for synonym in synonyms:
                if (synonym in all_vars_set and synonym not in nearest_vars
                        and synonym != target_var and synonym not in exclude_vars_set
                        and synonym not in formal_params_set):
                    nearest_vars.append(synonym)
                    if len(nearest_vars) >= k:
                        break

        nearest_dict[target_var] = nearest_vars[:k]

    return nearest_dict


def main():
    """主函数（与Word2Vec版本格式一致）"""
    print("开始基于WordNet的变量相似度计算...")

    # 读取所有变量列表
    print("读取所有变量列表...")
    try:
        data_all_var = pd.read_pickle(root + 'datasets/var_for_allCode.pkl')
        all_var_list = list(data_all_var['all vars'].tolist()[0])
        print(f"总变量数: {len(all_var_list)}")
    except Exception as e:
        print(f"读取所有变量列表失败: {e}")
        return

    # 读取每个代码的变量列表
    print("读取每个代码的变量列表...")
    try:
        data_every_var = pd.read_pickle(root + 'datasets/var_for_everyCode.pkl')
        every_var_list = data_every_var['variable'].tolist()
        print(f"代码数量: {len(every_var_list)}")
    except Exception as e:
        print(f"读取每个代码变量列表失败: {e}")
        return

    # 读取形式参数列表
    print("读取形式参数列表...")
    try:
        formal_parameter_data = pd.read_pickle(root + 'datasets/formalParameter_for_everyCode.pkl')
        formal_parameter_list = formal_parameter_data['formal_parameters'].tolist()
        print(f"形式参数列表数量: {len(formal_parameter_list)}")
    except Exception as e:
        print(f"读取形式参数列表失败: {e}")
        formal_parameter_list = [None] * len(every_var_list)

    # 计算最近邻（与Word2Vec版本逻辑一致）
    print("开始计算最近邻变量...")
    nearest_list = []
    count = 0

    for i, every_code_vars in enumerate(every_var_list):
        nearest_dict = {}

        # 获取当前代码的形式参数
        formal_params = formal_parameter_list[i] if i < len(formal_parameter_list) else None

        print(f'处理第 {count + 1} 个代码...')
        count += 1

        if i % 10 == 0:  # 每10个代码清理一次内存
            gc.collect()

        # 排除当前代码中的变量和形式参数（与Word2Vec版本一致）
        exclude_vars = every_code_vars.copy()
        if formal_params:
            exclude_vars.extend(formal_params)

        print(f"目标: {k} 个替换词")
        for var in every_code_vars:
            n_list = []

            if var not in all_var_list:
                print(f"变量 '{var}' 不在所有变量列表中，跳过")
                nearest_dict[var] = []
                continue

            # 计算最近邻
            nearest_dict_result = find_wordnet_nearest_variables(
                target_vars=[var],
                all_vars=all_var_list,
                k=k * 2,  # 多找一些，因为后面要筛选
                formal_params=formal_params,
                exclude_vars=exclude_vars
            )

            if var in nearest_dict_result:
                candidates = nearest_dict_result[var]

                # 去重并确保不包含排除的变量
                unique_candidates = []
                allcan_list = []  # 用于记录已经添加的候选词

                for candidate in candidates:
                    if (candidate not in allcan_list and
                            candidate not in exclude_vars and
                            candidate != var):
                        unique_candidates.append(candidate)
                        allcan_list.append(candidate)
                    if len(unique_candidates) >= k:
                        break

                n_list = unique_candidates[:k]
                found_count = len(n_list)
            else:
                found_count = 0
                n_list = []

            # 打印每个变量实际找到的替换词数量（与Word2Vec版本一致）
            print(f'"{var}" : {found_count}')
            nearest_dict[var] = n_list

        nearest_list.append(nearest_dict)
        print(f"代码 {i + 1} 处理完成，找到 {len(nearest_dict)} 个变量的最近邻")

    # 保存结果（与Word2Vec版本格式完全一致）
    print("保存结果...")
    try:
        os.makedirs('var_name', exist_ok=True)
        index = [i for i in range(len(nearest_list))]
        nearest_pd = pd.DataFrame({'id': index, 'nearest_k': nearest_list})
        save_path = f'./var_name/code_nearest_top{k}_wordnet_compatible.pkl'
        nearest_pd.to_pickle(save_path)
        print(f"结果已保存至: {save_path}")
    except Exception as e:
        print(f"保存结果失败: {e}")
        return

    # 强制清理内存
    gc.collect()

    # 统计信息
    total_vars = sum(len(vars) for vars in every_var_list)
    vars_with_neighbors = sum(1 for nearest_dict in nearest_list
                              for neighbors in nearest_dict.values() if neighbors)

    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"总代码数量: {len(every_var_list)}")
    print(f"总变量数量: {total_vars}")
    print(f"成功找到最近邻的变量数量: {vars_with_neighbors}")
    print(f"最近邻数量k: {k}")
    print("使用的模型: WordNet语义相似度")
    print("=" * 50)


if __name__ == '__main__':
    main()
