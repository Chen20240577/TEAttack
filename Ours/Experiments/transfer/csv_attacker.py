# -*- coding: utf-8 -*-
import csv
import sys

from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../../')
sys.path.append('../../../python_parser')

from utils import CodeDataset, GraphCodeDataset, CodePairDataset
from load import load


class Attacker:

    def __init__(self, args, model, tokenizer) -> None:
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

    # def csv_attack(self, csv_path, args, transfer_recoder):
    #
    #     csv.field_size_limit(sys.maxsize)
    #
    #     # 打开 CSV 文件
    #     with open(csv_path, mode='r', encoding='utf-8') as csv_file:
    #         reader = csv.DictReader(csv_file)
    #         rows = list(reader)  # 读取所有行
    #
    #     success_attack = 0  # 成功攻击次数
    #     total_cnt = 0  # 总对抗示例数量
    #     invalid_cnt = 0  # 无效对抗示例数量
    #     original_fail_cnt = 0  # 原始预测错误的样本数量
    #
    #     # 遍历每一行
    #     for row in tqdm(rows, desc="验证对抗示例"):
    #         # 读取对抗代码和其他信息
    #         index = row.get('Index', '').strip()
    #         adv_code = row.get('Adversarial Code', '').strip()  # 获取对抗代码，并去除空白字符
    #         true_label = row.get('True Label', '').strip()  # 真实标签
    #         orig_code = row.get('Original Code', '').strip()
    #         orig_label = row.get('Original Prediction', '').strip()  # 原始预测标签
    #
    #         # 检查数据是否有效
    #         if not adv_code or not true_label or not orig_code or not orig_label:
    #             invalid_cnt += 1
    #             continue  # 跳过无效数据
    #         # if not true_label or not orig_code or not orig_label:
    #         #     invalid_cnt += 1
    #         #     continue  # 跳过无效数据
    #
    #         try:
    #             true_label = int(true_label)
    #             orig_label = int(orig_label)
    #         except ValueError:
    #             invalid_cnt += 1
    #             continue  # 跳过无法转换为整数的标签
    #
    #         total_cnt += 1
    #
    #         orig_example = []
    #         orig_example.append(
    #             load.conduct_example(
    #                 args, orig_code, true_label,
    #                 self.tokenizer, self.code_max_len,
    #                 self.front_token, self.behind_token, None
    #             )
    #         )
    #         orig_dataset = self.Dataset(orig_example)
    #
    #         orig_logits, orig_preds = self.target_model.get_results(orig_dataset, self.args.batch_size)
    #         pre_label = orig_preds[0]
    #         # if pre_label== true_label:
    #         #     success_pre += 1
    #         #     continue
    #         if pre_label != true_label:
    #             original_fail_cnt += 1
    #             continue
    #
    #         # 构造示例（假设示例格式与之前相同）
    #         adv_example = []
    #         adv_example.append(
    #             load.conduct_example(
    #                 args, orig_code, true_label,
    #                 self.tokenizer, self.code_max_len,
    #                 self.front_token, self.behind_token, adv_code
    #             )
    #         )
    #         adv_dataset = self.Dataset(adv_example)
    #         # 使用目标模型对对抗示例进行预测
    #         logits, preds = self.target_model.get_results(adv_dataset, self.args.batch_size)
    #         temp_label = preds[0]
    #
    #         # 统计攻击结果
    #         if temp_label != true_label:
    #             transfer_recoder.write(index, orig_code, adv_code, True)
    #             success_attack += 1
    #         else:
    #             transfer_recoder.write(index, orig_code, adv_code, False)
    #
    #     # 提取 CSV 文件名和基础模型名称
    #     csv_name = csv_path.split('/')[-1].split('.')[0]  # 提取文件名，去掉路径和扩展名
    #     base_model_name = args.model_type  # 提取基础模型名称
    #     # 输出统计信息
    #     print(f"对抗示例验证完成！")
    #     print(f"对抗示例来源: {csv_name}")
    #     print(f"攻击的模型: {base_model_name}")
    #     print(f"对抗示例总量: {total_cnt}")
    #     print(f"原始预测错误的样本数量: {original_fail_cnt}")
    #     print(f"有效对抗示例总量: {total_cnt - original_fail_cnt}")
    #     print(f"成功攻击次数: {success_attack}")
    #     if total_cnt > 0:
    #         denominator = total_cnt - original_fail_cnt
    #         success_rate = success_attack / denominator if denominator != 0 else 0.0
    #         print(f"对抗示例攻击成功率: {success_rate:.4f}")
    #     else:
    #         print("未找到有效的对抗示例。")
    #     print(f"无效样本数量: {invalid_cnt}")

    def csv_attack(self, csv_path, args, transfer_recoder):

        csv.field_size_limit(sys.maxsize)

        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        success_attack = 0
        total_cnt = 0
        invalid_cnt = 0
        original_fail_cnt = 0

        # === 第一步：读取并预处理所有样本 ===
        valid_samples = []

        for row in tqdm(rows, desc="读取样本"):
            index = row.get('Index', '').strip()
            adv_code = row.get('Adversarial Code', '').strip()
            true_label = row.get('True Label', '').strip()
            orig_code = row.get('Original Code', '').strip()

            if not adv_code or not true_label or not orig_code:
                invalid_cnt += 1
                continue

            try:
                true_label = int(true_label)
            except ValueError:
                invalid_cnt += 1
                continue

            total_cnt += 1
            valid_samples.append({
                'index': index,
                'orig_code': orig_code,
                'adv_code': adv_code,
                'true_label': true_label,
            })

        if not valid_samples:
            print("未找到任何有效样本。")
            return

        # === 第二步：批量验证原始代码的预测 ===
        print("\n\n批量验证原始代码预测...")
        orig_examples = []

        for sample in valid_samples:
            ex = load.conduct_example(
                args, sample['orig_code'], sample['true_label'],
                self.tokenizer, self.code_max_len,
                self.front_token, self.behind_token, None
            )
            orig_examples.append(ex)

        orig_dataset = self.Dataset(orig_examples)
        orig_logits, orig_preds = self.target_model.get_results(orig_dataset, self.args.batch_size)

        # 筛选出原始预测正确的样本
        correctly_predicted_samples = []

        for i, (pred, sample) in enumerate(zip(orig_preds, valid_samples)):
            if pred == sample['true_label']:
                # print("Pred:", orig_preds[0])
                # print("Logits:", orig_logits[0])

                correctly_predicted_samples.append(sample)
            else:
                original_fail_cnt += 1

        if not correctly_predicted_samples:
            print("没有原始预测正确的样本。")
            self._print_stats(csv_path, args, total_cnt, invalid_cnt, original_fail_cnt, 0, 0)
            return

        # === 第三步：批量验证对抗代码 ===
        print("批量验证对抗代码...")
        adv_examples = []

        for sample in correctly_predicted_samples:
            ex = load.conduct_example(
                args, sample['orig_code'], sample['true_label'],
                self.tokenizer, self.code_max_len,
                self.front_token, self.behind_token, sample['adv_code']
            )
            adv_examples.append(ex)

        adv_dataset = self.Dataset(adv_examples)
        adv_logits, adv_preds = self.target_model.get_results(adv_dataset, self.args.batch_size)
        # print("Pred:", adv_preds[0])
        # print("Logits:", adv_logits[0])
        # === 第四步：统计结果 ===
        for i, (pred, sample) in enumerate(zip(adv_preds, correctly_predicted_samples)):
            if pred != sample['true_label']:
                transfer_recoder.write(sample['index'], sample['orig_code'],
                                       sample['adv_code'], True)
                success_attack += 1
            else:
                transfer_recoder.write(sample['index'], sample['orig_code'],
                                       sample['adv_code'], False)

        # === 第五步：输出统计信息 ===
        self._print_stats(csv_path, args, total_cnt, invalid_cnt,
                          original_fail_cnt, success_attack, len(correctly_predicted_samples))

    def _print_stats(self, csv_path, args, total_cnt, invalid_cnt,
                     original_fail_cnt, success_attack, valid_adv_samples_count):
        """输出统计信息的辅助函数"""
        csv_name = csv_path.split('/')[-1].split('.')[0]
        base_model_name = args.model_type

        print(f"\n{'=' * 50}")
        print(f"对抗示例验证完成！")
        print(f"{'=' * 50}")
        print(f"对抗示例来源: {csv_name}")
        print(f"攻击的模型: {base_model_name}")
        print(f"对抗示例总量: {total_cnt}")
        print(f"原始预测错误的样本数量: {original_fail_cnt}")
        print(f"有效对抗示例总量: {valid_adv_samples_count}")
        print(f"成功攻击次数: {success_attack}")

        if valid_adv_samples_count > 0:
            success_rate = success_attack / valid_adv_samples_count
            print(f"对抗示例攻击成功率: {success_rate:.4f} ({success_attack}/{valid_adv_samples_count})")
        else:
            print("没有有效的对抗示例用于计算攻击成功率。")

        print(f"无效样本数量: {invalid_cnt}")
        print(f"{'=' * 50}")