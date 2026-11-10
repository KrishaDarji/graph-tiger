import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import Adafactor as HFAdafactor


def _rms(t):
    return t.norm(2) / (t.numel() ** 0.5)


class Adafactor(torch.optim.Optimizer):

    def __init__(self, params, lr=0.01, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8):
        super().__init__(params, dict(lr=lr, eps=eps, clip_threshold=clip_threshold, decay_rate=decay_rate))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    if grad.dim() >= 2:
                        state["row"] = torch.zeros(grad.shape[:-1], device=grad.device)
                        state["col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device)
                    else:
                        state["sq"] = torch.zeros_like(grad)
                state["step"] += 1
                beta2t = 1.0 - state["step"] ** group["decay_rate"]
                scale = _rms(p).clamp(min=group["eps"][1]) * group["lr"]  # tensor, no sync
                sq = grad * grad + group["eps"][0]
                if grad.dim() >= 2:
                    state["row"].mul_(beta2t).add_(sq.mean(dim=-1), alpha=1.0 - beta2t)
                    state["col"].mul_(beta2t).add_(sq.mean(dim=-2), alpha=1.0 - beta2t)
                    r = (state["row"] / state["row"].mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
                    update = (r * state["col"].unsqueeze(-2).rsqrt()) * grad
                else:
                    state["sq"].mul_(beta2t).add_(sq, alpha=1.0 - beta2t)
                    update = state["sq"].rsqrt() * grad
                update = update / (_rms(update) / group["clip_threshold"]).clamp(min=1.0)
                p.add_(-(update * scale).to(p.dtype))
        return loss


def build_optimizer(parameters, name="adafactor", lr=0.01, weight_decay=0.0):
    if name == "adafactor":
        return Adafactor(parameters, lr=lr)  # T5X settings; sync-free, matches HF (tested)
    if name == "adafactor_hf":
        return HFAdafactor(parameters, lr=lr, scale_parameter=True, relative_step=False,
                           warmup_init=False, clip_threshold=1.0, decay_rate=-0.8, beta1=None,
                           weight_decay=0.0)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def constant_then_inverse_sqrt(optimizer, constant_steps=10_000):

    def multiplier(step):
        return min(1.0, math.sqrt(constant_steps / max(step, 1)))

    return LambdaLR(optimizer, multiplier)
