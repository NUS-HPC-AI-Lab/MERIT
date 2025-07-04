"""MERIT optimizer."""

import torch
from torch.optim import Optimizer

class MERIT(Optimizer):
    r"""Implements MERIT algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.

    """

    def __init__(
            self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        super(MERIT, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lamb does not support sparse gradients, consider SparseAdam instad."
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Apply bias to lr to avoid broadcast.
                step_size = group[
                    "lr"
                ]  # * math.sqrt(bias_correction2) / bias_correction1

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"])

                if len(grad.shape) >= 2:
                    weight_abs = p.data.abs()
                    adam_abs = adam_step.abs()
                    weight_ratio = weight_abs.max() / adam_abs.max()

                    # l1 norm
                    row_weight_norm = weight_abs.amax(dim=1, keepdim=True).expand_as(grad)
                    col_weight_norm = weight_abs.amax(dim=0, keepdim=True).expand_as(grad)

                    row_adam_norm = adam_abs.amax(dim=1, keepdim=True).expand_as(grad)
                    col_adam_norm = adam_abs.amax(dim=0, keepdim=True).expand_as(grad)

                    row_trust_ratio = row_weight_norm / row_adam_norm
                    col_trust_ratio = col_weight_norm / col_adam_norm

                    idx = (row_weight_norm == 0) | (row_adam_norm == 0) | (col_weight_norm == 0) | (col_adam_norm == 0)

                    trust_ratio = torch.max(col_trust_ratio, row_trust_ratio)

                    trust_ratio = trust_ratio.clamp(min=weight_ratio)
                    trust_ratio[idx] = 1

                    adam_step.mul_(trust_ratio)
                    adam_step_clip = adam_step.abs().clamp(None, 1)
                    adam_step = adam_step.sign() * adam_step_clip

                    p.data.add_(adam_step, alpha=-step_size)


                else:
                    weight_max = p.data.abs().max()
                    adam_max = adam_step.abs().max()

                    if weight_max == 0 or adam_max == 0:
                        trust_ratio = 1
                    else:
                        trust_ratio = weight_max / adam_max

                    adam_step.mul_(trust_ratio)
                    adam_step_clip = adam_step.abs().clamp(None, 1)
                    adam_step = adam_step.sign() * adam_step_clip

                    p.data.add_(adam_step, alpha=-step_size)

        return loss
