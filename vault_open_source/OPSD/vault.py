import math
from collections import deque

import numpy as np
import torch


class VaultWeighter:
    """Sequence-level VAULT weight controller.

    The closed-form raw weight is
        s * (1 - exp(-k / kappa)) / (s * (1 + (k / kappa)^2) + (1 - s) * c).
    Ablation modes change only the raw valuation function; normalization and
    clipping stay shared for fair comparisons.
    """

    def __init__(
        self,
        window_size=256,
        min_valid=16,
        init_kappa=0.3,
        init_norm=0.3,
        norm_alpha=0.01,
        c=1.0,
        s_min=0.05,
        valid_w_min=0.05,
        valid_w_max=2.0,
        censored_w_min=0.0,
        censored_w_max=0.1,
        kappa_floor=1e-4,
        survival_mode="constant",
        bell_sigma=0.5,
        clip_mode="by_validity",
        kappa_update_mode="valid",
        weight_mode="vault",
        fixed_censored_weight=0.5,
        eps=1e-8,
    ):
        self.k_queue = deque(maxlen=int(window_size))
        self.min_valid = int(min_valid)
        self.kappa = float(init_kappa)
        self.kappa_floor = float(kappa_floor)
        self.norm_ema = float(init_norm)
        self.norm_alpha = float(norm_alpha)
        self.c = float(c)
        self.s_min = float(s_min)
        self.valid_w_min = float(valid_w_min)
        self.valid_w_max = float(valid_w_max)
        self.censored_w_min = float(censored_w_min)
        self.censored_w_max = float(censored_w_max)
        self.survival_mode = str(survival_mode)
        self.bell_sigma = float(bell_sigma)
        self.clip_mode = str(clip_mode)
        self.kappa_update_mode = str(kappa_update_mode)
        self.weight_mode = str(weight_mode)
        self.fixed_censored_weight = float(fixed_censored_weight)
        self.eps = float(eps)
        if self.survival_mode not in {"constant", "bell"}:
            raise ValueError(f"Unsupported VAULT survival_mode: {self.survival_mode}")
        if self.clip_mode not in {"by_validity", "global"}:
            raise ValueError(f"Unsupported VAULT clip_mode: {self.clip_mode}")
        if self.kappa_update_mode not in {"valid", "all"}:
            raise ValueError(f"Unsupported VAULT kappa_update_mode: {self.kappa_update_mode}")
        if self.weight_mode not in {"vault", "hard_filter", "fixed_penalty", "sweet_spot", "survival_only"}:
            raise ValueError(f"Unsupported VAULT weight_mode: {self.weight_mode}")
        if self.fixed_censored_weight < 0:
            raise ValueError("VAULT fixed_censored_weight must be nonnegative.")
        if self.bell_sigma <= 0:
            raise ValueError("VAULT bell_sigma must be positive.")

    @torch.no_grad()
    def update_kappa(self, k, valid):
        if self.kappa_update_mode == "all":
            update_mask = torch.ones_like(valid.detach().bool())
        else:
            update_mask = valid.detach().bool()
        valid_k = k.detach().float()[update_mask].cpu().tolist()
        self.update_kappa_from_values(valid_k)

    def update_kappa_from_values(self, valid_k_values):
        for value in valid_k_values:
            if value is not None and math.isfinite(float(value)) and float(value) > 0:
                self.k_queue.append(float(value))
        if len(self.k_queue) >= self.min_valid:
            median = float(np.median(np.array(self.k_queue, dtype=np.float64)))
            self.kappa = max(median, self.kappa_floor)

    @torch.no_grad()
    def survival_prob(self, k, valid):
        k = k.detach().float()
        valid = valid.detach().bool()
        x = k / (self.kappa + self.eps)
        if self.survival_mode == "bell":
            unobserved = torch.exp(-((x - 1.0) ** 2) / (2.0 * self.bell_sigma * self.bell_sigma))
            unobserved = unobserved.clamp(self.s_min, 1.0)
        else:
            unobserved = torch.full_like(x, self.s_min)
        return torch.where(valid, torch.ones_like(x), unobserved)

    @torch.no_grad()
    def raw_weights(self, k, valid):
        k = k.detach().float()
        valid = valid.detach().bool()
        x = k / (self.kappa + self.eps)
        learnability = 1.0 - torch.exp(-x)
        valid_risk = 1.0 + x * x
        survival = self.survival_prob(k, valid)
        if self.weight_mode == "hard_filter":
            return torch.where(valid, torch.ones_like(x), torch.zeros_like(x))
        if self.weight_mode == "fixed_penalty":
            censored = torch.full_like(x, self.fixed_censored_weight)
            return torch.where(valid, torch.ones_like(x), censored)
        if self.weight_mode == "sweet_spot":
            return learnability / (valid_risk + self.eps)
        if self.weight_mode == "survival_only":
            return survival
        return survival * learnability / (
            survival * valid_risk + (1.0 - survival) * self.c + self.eps
        )

    def update_norm_from_mean(self, valid_raw_mean):
        if valid_raw_mean is None:
            return
        value = float(valid_raw_mean)
        if not math.isfinite(value):
            return
        self.norm_ema = (1.0 - self.norm_alpha) * self.norm_ema + self.norm_alpha * value

    @torch.no_grad()
    def update_norm(self, raw_w, valid):
        valid = valid.detach().bool()
        if valid.any():
            self.update_norm_from_mean(float(raw_w[valid].mean().item()))

    @torch.no_grad()
    def normalize_and_clip(self, raw_w, valid):
        valid = valid.detach().bool()
        w = raw_w / (self.norm_ema + self.eps)
        if self.clip_mode == "global":
            return w.clamp(self.valid_w_min, self.valid_w_max).detach()
        w_valid = w.clamp(self.valid_w_min, self.valid_w_max)
        w_censored = w.clamp(self.censored_w_min, self.censored_w_max)
        return torch.where(valid, w_valid, w_censored).detach()

    @torch.no_grad()
    def compute_weight(self, k, valid):
        raw_w = self.raw_weights(k, valid)
        self.update_norm(raw_w, valid)
        return self.normalize_and_clip(raw_w, valid), raw_w.detach()
