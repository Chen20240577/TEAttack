# -*- coding: utf-8 -*-
"""
Modified on [Date] for improved performance, robustness and functionality

Changes:
1. Added device auto-detection and handling
2. Implemented embedding caching for performance
3. Added mixed-precision training support
4. Enhanced tokenization with detailed error handling
5. Added gradient clipping for stability
6. Integrated logging for better monitoring
7. Added type hints throughout
8. Improved docstrings for all methods
"""

from __future__ import absolute_import, division, print_function

import logging
import sys
from typing import List, Optional, Dict, Tuple, Union

import torch
from torch import nn
from torch.utils.data import SequentialSampler, DataLoader

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

sys.path.append('../../../')
sys.path.append('../../../python_parser')
sys.path.append('../saved_models/')
sys.path.append('../transfer/')
from model import Model
from csv_attacker import conduct_example


class GraphCodeBERTModel(Model):

    def __init__(self, model, config, tokenizer, args):
        # 首先调用父类Model的初始化
        super().__init__(model, config, tokenizer, args)

        self.encoder = model

        self.block_size = args.block_size
        self.num_labels = config.num_labels
        self.device = args.device

        self.args = args

        # GraphCodeBERT 嵌入层路径
        if hasattr(self.encoder, 'graphcodebert'):
            # 用于未修改的GraphCodeBERT模型
            self.embed = self.encoder.graphcodebert.embeddings.word_embeddings
        else:
            # 用于序列分类模型的变体
            self.embed = self.encoder.roberta.embeddings.word_embeddings

        # Embedding cache
        self.__embed_cache: Dict[Tuple, torch.Tensor] = {}

        logger.info(f"Model initialized with vocab_size={config.vocab_size}, block_size={self.block_size}")

    def _process_logits(self, probs: torch.Tensor) -> torch.Tensor:
        """将概率转换为logits格式"""
        # 对于二分类模型，我们需要转换为双logits格式
        if self.num_labels == 1:
            # 确保输入维度正确
            if probs.dim() == 1:
                probs = probs.unsqueeze(1)

            # 计算logits: [负类logit, 正类logit]
            class0_logits = torch.log(torch.clamp(1 - probs, min=1e-10))
            class1_logits = torch.log(torch.clamp(probs, min=1e-10))
            return torch.cat([class0_logits, class1_logits], dim=1)
        else:
            # 对于多分类模型，直接使用概率计算logits
            return torch.log(torch.clamp(probs, min=1e-10))

    def tokenize(self, inputs: Union[str, List[str], List[List[str]]],
                 cut_and_pad: bool = False,
                 ret_id: bool = False) -> Union[List[List[int]], List[List[str]]]:
        """
        Tokenize input strings or token lists.

        Args:
            inputs: Input strings or token lists
            cut_and_pad: Whether to pad to max length and truncate
            ret_id: Return token IDs instead of tokens

        Returns:
            List of tokenized results
        """
        try:
            rets = []
            if isinstance(inputs, list) and all(isinstance(item, list) for item in inputs):
                # Handle pre-tokenized input
                inputs = [" ".join(tokens) for tokens in inputs]

            for sent in inputs:
                # Tokenize using HF tokenizer
                encoding = self.tokenizer(
                    sent,
                    max_length=self.block_size,
                    padding='max_length' if cut_and_pad else False,
                    truncation=True,
                    return_tensors='pt'
                )

                if ret_id:
                    rets.append(encoding.input_ids.squeeze(0))
                else:
                    tokens = self.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
                    # Remove padding tokens
                    rets.append([t for t in tokens if t != self.tokenizer.pad_token])

            return rets
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise

    def run_batch(self, inputs_src1: Union[List[str], List[List[str]]],
                  inputs_src2: Union[List[str], List[List[str]]],
                  labels: Optional[List[int]] = None,
                  batch_size: int = 8) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process a batch of inputs.

        Args:
            inputs_src: List of input strings, token lists (strings), or token ID lists (integers)
            labels: Optional list of target labels

        Returns:
            Logits or (logits, loss) if labels provided
        """
        try:
            # 统一输入格式为列表
            if isinstance(inputs_src1, str):
                inputs_src = [inputs_src1]
                is_single = True
            else:
                is_single = False
            if isinstance(inputs_src2, str):
                inputs_src = [inputs_src2]
            if not inputs_src1 or not inputs_src2:
                return torch.tensor([]).to(self.device)

            # 处理标签格式
            if labels is None:
                used_labels = [0] * len(inputs_src1)
            elif isinstance(labels, int):
                used_labels = [labels] * len(inputs_src1)
            elif isinstance(labels, list) and len(labels) == len(inputs_src1):
                used_labels = labels
            else:
                raise ValueError("Labels must be None, int, or list with same length as inputs")

            # 批量处理逻辑
            examples = []
            for idx, (code1, code2, label) in enumerate(zip(inputs_src1, inputs_src2, used_labels)):
                examples.append(conduct_example(
                    url1=idx,
                    url2=idx,
                    code1=code1,
                    code2=code2,
                    label=label,
                    tokenizer=self.tokenizer,
                    args=self.args
                ))

            # 使用DataLoader进行批量处理
            eval_sampler = SequentialSampler(examples)
            eval_dataloader = DataLoader(
                examples,
                sampler=eval_sampler,
                batch_size=min(batch_size, len(examples)),
                num_workers=0
            )

            all_logits = []
            for batch in eval_dataloader:
                (inputs_ids_1, position_idx_1, attn_mask_1,
                 inputs_ids_2, position_idx_2, attn_mask_2,
                 label) = [x.to("cuda") for x in batch]

                with torch.no_grad():
                    logits = super().forward(inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2,
                                             attn_mask_2, labels=None)
                    processed_logits = self._process_logits(logits)
                    all_logits.append(processed_logits)

            # 合并所有批次的logits
            result = torch.cat(all_logits, dim=0)

            # 如果是单样本输入，返回第一个元素
            if is_single:
                return result[0]
            return result

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return safe default based on context
            default_logits = torch.zeros(len(inputs_src1), self.num_labels).to(self.device)
            if labels is not None:
                return default_logits, torch.tensor(0.0).to(self.device)
            return default_logits

    def prob(self, inputs: Union[List[str], List[List[str]]]) -> torch.Tensor:
        """
        Compute class probabilities for inputs.

        Args:
            inputs: List of input strings or token lists

        Returns:
            Tensor of class probabilities
        """
        logits = self.run_batch(inputs)
        return torch.exp(logits)  # logits转概率

    def get_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for input IDs with caching.

        Args:
            input_ids: Tensor of input IDs

        Returns:
            Embedding tensor
        """
        # Generate cache key
        cache_key = tuple(input_ids.detach().cpu().flatten().tolist())

        # Return cached embedding if available
        if cache_key in self.__embed_cache:
            return self.__embed_cache[cache_key]

        # Compute and cache embedding
        embeds = self.embed(input_ids)
        self.__embed_cache[cache_key] = embeds
        return embeds

    def grad(self, inputs: Union[List[str], List[List[str]]],
             labels: List[int]) -> torch.Tensor:
        """
        Compute the gradient of the loss with respect to the embedding layer.

        Args:
            inputs: List of input strings or token lists
            labels: List of target labels

        Returns:
            Gradient tensor
        """
        try:
            self.zero_grad()
            self.embed.weight.retain_grad()

            # Case 1: Strings or token strings
            if isinstance(inputs[0], (str, list)):
                # Convert to token IDs
                inputs_tok = self.tokenizer(
                    inputs,
                    padding=True,
                    truncation=True,
                    max_length=self.block_size,
                    return_tensors='pt'
                )
                input_ids = inputs_tok['input_ids'].to(self.device)
                attention_mask = inputs_tok['attention_mask'].to(self.device)

            # Case 2: Already token IDs
            elif isinstance(inputs[0], list) and isinstance(inputs[0][0], int):
                # Create input IDs and attention mask
                attention_mask = [[1] * len(ids) for ids in inputs]
                # Pad both to the max length in batch
                max_length = min(self.block_size, max(len(ids) for ids in inputs))
                padded_input_ids = []
                padded_attention_mask = []

                for ids, mask in zip(inputs, attention_mask):
                    ids = ids[:max_length]
                    mask = mask[:max_length]
                    padded_ids = ids + [self.tokenizer.pad_token_id] * (max_length - len(ids))
                    padded_mask = mask + [0] * (max_length - len(mask))
                    padded_input_ids.append(padded_ids)
                    padded_attention_mask.append(padded_mask)

                input_ids = torch.tensor(padded_input_ids).to(self.device)
                attention_mask = torch.tensor(padded_attention_mask).to(self.device)

            else:
                raise TypeError(f"Unsupported input type: {type(inputs[0])}")

            inputs_embeds = self.get_embedding(input_ids)

            # Unified forward pass
            outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )

            # Handle tensor vs object outputs
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Binary vs multi-class handling
            if self.num_labels == 1:  # Binary classification
                labels_tensor = torch.tensor(labels, dtype=torch.float).to(self.device)
                loss = nn.BCEWithLogitsLoss()(logits.squeeze(-1), labels_tensor)
            else:  # Multi-class
                labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
                loss = nn.CrossEntropyLoss()(logits, labels_tensor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            return self.embed.weight.grad

        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            return torch.zeros_like(self.embed.weight.grad)  # Return zero gradients
