# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from opt_einsum import contract
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)


logger = logging.getLogger(__name__)


class DCSGDPOptimizer(DPOptimizer):
    """
    :class:~opacus.optimizers.optimizer.DPOptimizer that implements
    adaptive clipping strategy
    https://arxiv.org/pdf/1905.03871.pdf
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        histogram_std: float = 6.0,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        batchsize_train: int = 256,
        dimension: int = 11181642,
        percentile: float = 0.3,
        stride: float = 2.0,
        bin_cnt: int = 20,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        # dimension
        # resnet18 11181642
        # resnet34 21289802
        # self.historgram_std = histogram_std
        self.historgram_std = histogram_std
        self.batchsize_train = batchsize_train
        self.dimension = dimension
        self.percentile = percentile
        self.timer = 0
        self.stride = 1.0
        self.bin_cnt = bin_cnt
        self.noise_multiplier = (
            self.noise_multiplier ** (-2) - (self.historgram_std) ** (-2)
        ) ** (-1 / 2)
        self.sample_size = 0
        self.unclipped_num = 0
        self.norm_stack = []

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients, self.sample_size and self.unclipped_num
        """
        super().zero_grad(set_to_none)

        self.sample_size = 0
        self.unclipped_num = 0

    def clip_and_accumulate(self):


        extended_grad_samples = []
        per_sample_norms_all = []

        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = self._get_flat_grad_sample(p)  # [B, D]
            #print(f"[DEBUG] grad_sample shape: {grad_sample.shape}")  # 应该是 (B, D)
            grad_sample = grad_sample.view(grad_sample.shape[0], -1)
            B, D = grad_sample.shape
            norms = grad_sample.norm(2, dim=1)  # [B]
            per_sample_norms_all.append(norms)

            residuals = (norms - self.max_grad_norm).clamp(min=0.0)  # [B]，如果超过C，就记录下超出的部分residual
            r_per_dim = self.max_grad_norm / 5  # 每个残差槽最多填入 C/5

            # 初始化扩展维度
            extra_dims = torch.zeros(B, 5, device=grad_sample.device)
            residuals_copy = residuals.clone()
            for i in range(5):
                fill = torch.min(residuals_copy, torch.full_like(residuals_copy, r_per_dim))
                extra_dims[:, i] = fill
                residuals_copy -= fill

            # 拼接原始梯度和扩展维度
            extended_grad = torch.cat([grad_sample, extra_dims], dim=1)  # [B, D+5]
            extended_grad_samples.append(extended_grad)
            # 记录所有样本的5个残差槽信息
            self.residual_slots = torch.cat(
                [ext[:, -5:] for ext in extended_grad_samples], dim=0  # [B, 5]
            ).detach().cpu()



            _mark_as_processed(p.grad_sample)

        # 统一计算剪裁因子（基于拼接后的向量范数）
        all_per_sample_norms = []
        for ext_grad in extended_grad_samples:
            norm = ext_grad.norm(2, dim=1)  # 包含扩展维度的范数
            all_per_sample_norms.append(norm)
        all_per_sample_norms = torch.cat(all_per_sample_norms, dim=0)
        self.norm_stack.append(all_per_sample_norms.detach().cpu())

        # 仅对原始梯度部分进行剪裁和累积
        for p, ext_grad in zip(self.params, extended_grad_samples):
            grad_part = ext_grad[:, :-5]  # [B, D]，原始梯度
            norms = ext_grad.norm(2, dim=1)  # [B]，含扩展维度
            clip_factors = (self.max_grad_norm / (norms + 1e-6)).clamp(max=1.0)  # [B]
            clipped = contract("i,i...->...", clip_factors, grad_part)  # [D]

            # 累积梯度
            if p.summed_grad is not None:
                p.summed_grad += clipped
            else:
                p.summed_grad = clipped


    def add_noise(self):
        super().add_noise()

    def set_epoch(self, epoch):  # 加在 DCSGDPOptimizer 类中
        self.current_epoch = epoch

    def update_max_grad_norm_from_extensions(self):
        print(f"[DCSGDP] Updated C to {self.max_grad_norm:.4f}")

        if not hasattr(self, "residual_slots"):
            return

        # Step 1: 分析 residual_slots
        slot_use = (self.residual_slots > 0).float()
        slot_counts = slot_use.sum(dim=0)
        total_samples = slot_use.shape[0]
        slot_weights = torch.tensor([1, 4, 9, 16, 25], dtype=torch.float32)
        weighted_sum = (slot_counts @ slot_weights).item()
        avg_slot_score = weighted_sum / (total_samples + 1e-6)

        # Step 2: 动态确定 target_score（你可以微调 base 值）
        base_target = 10.0
        if hasattr(self, "current_epoch") and self.current_epoch < 5:
            target_score = base_target * 2.0  # 前期允许更激进
        else:
            target_score = base_target

        # Step 3: 动态 factor 调整，响应更敏感
        delta = (avg_slot_score - target_score) / (target_score + 1e-6)
        factor = 1 + 0.3 * delta  # 比例变大，更敏感

        # Step 4: 前期更激进，后期更平滑
        if hasattr(self, "current_epoch") and self.current_epoch < 5:
            alpha = 0.3
        else:
            alpha = 0.9

        # Step 5: 更新 C 值
        old_C = self.max_grad_norm
        new_c = self.max_grad_norm * factor
        self.max_grad_norm = alpha * self.max_grad_norm + (1 - alpha) * new_c

        # 限制范围
        self.max_grad_norm = max(min(self.max_grad_norm, 10.0), 0.01)

        # Step 6: 日志记录
        logger.info(
            f"[DCSGDP] Epoch {getattr(self, 'current_epoch', '?')}: "
            f"Adjusted C from {old_C:.4f} → {self.max_grad_norm:.4f} "
            f"(Avg Slot Score = {avg_slot_score:.2f}, Target = {target_score:.2f}, α = {alpha:.2f})"
        )

        self.current_clip = self.max_grad_norm
        del self.residual_slots


    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        loss = super().pre_step(closure)
        self.update_max_grad_norm_from_extensions()
        return loss