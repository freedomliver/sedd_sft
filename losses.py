import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps

        sigma, dsigma = noise(t)

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):

    def optimize_fn(optimizer,
                    scaler,
                    params,
                    step,
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum

            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()

                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn


def _sample_t(batch_size, device, sampling_eps, sigma_max=None, t_min=0.0, t_max=None):
    if sigma_max is None:
        upper = 1.0 - sampling_eps
    else:
        eps_noise = 1e-3
        upper = (1 - math.exp(-sigma_max)) / (1 - eps_noise)
        upper = min(upper, 1.0 - sampling_eps)
    if t_max is not None:
        upper = min(upper, float(t_max))
    lower = max(float(t_min), sampling_eps)
    lower = min(lower, upper)
    return (upper - lower) * torch.rand(batch_size, device=device) + lower


def get_response_only_sft_loss_fn(
    noise,
    graph,
    sampling_eps=1e-3,
    t_min=0.0,
    t_max=None,
    high_t_frac=0.0,
    high_t_min=0.75,
    high_t_max=None,
):
    """Response-only DWDSE loss for conditional SEDD SFT.

    The prompt and padding tokens stay clean. Only answer tokens are forward-noised,
    and only answer tokens contribute to the score-entropy objective.
    """

    def loss_fn(
        model,
        batch,
        answer_mask,
        pad_mask=None,
        sigma_max=None,
        prompt_len=None,
        answer_window_len=None,
    ):
        batch = batch.long()
        B = batch.shape[0]
        if pad_mask is None:
            pad_mask = torch.ones_like(answer_mask, dtype=torch.bool)

        loss_mask = answer_mask.bool() & pad_mask.bool()
        if not torch.any(loss_mask):
            raise ValueError("response-only SFT batch has no answer tokens")

        t = _sample_t(B, batch.device, sampling_eps, sigma_max, t_min=t_min, t_max=t_max)
        if high_t_frac > 0:
            high_t = _sample_t(
                B,
                batch.device,
                sampling_eps,
                sigma_max,
                t_min=high_t_min,
                t_max=high_t_max if high_t_max is not None else t_max,
            )
            choose_high = torch.rand(B, device=batch.device) < high_t_frac
            t = torch.where(choose_high, high_t, t)
        sigma, dsigma = noise(t)

        noisy_batch = graph.sample_transition(batch, sigma[:, None])
        perturb_mask = loss_mask
        if prompt_len is not None:
            positions = torch.arange(batch.shape[1], device=batch.device).unsqueeze(0)
            start = prompt_len.to(batch.device).long().unsqueeze(1)
            perturb_mask = positions >= start
            if answer_window_len is not None:
                perturb_mask = perturb_mask & (positions < start + int(answer_window_len))
        perturbed = torch.where(perturb_mask, noisy_batch, batch)

        log_score_fn = mutils.get_score_fn(model, train=True, sampling=False)
        log_score = log_score_fn(perturbed, sigma)
        token_loss = graph.score_entropy(log_score, sigma[:, None], perturbed, batch)

        weighted = dsigma[:, None] * token_loss * loss_mask.to(token_loss.dtype)
        denom = loss_mask.sum(dim=-1).clamp_min(1).to(token_loss.dtype)
        return weighted.sum(dim=-1) / denom

    return loss_fn


def get_answer_all_mask_ce_loss_fn(
    noise,
    graph,
    t_values=(1e-4, 1e-2, 1e-1, 5e-1, 1.0),
    cropped=False,
    include_window_eos=False,
    answer_weight=1.0,
):
    """Auxiliary CE for small-sample overfit from prompt + fully masked answer.

    This does not replace DWDSE. It directly teaches the sampling start state:
    prompt tokens are clean/clamped, every answer token is graph.absorb/mask, and
    the model must rank the original answer token highest at each answer position.
    """

    t_values = tuple(float(t) for t in t_values)

    def loss_fn(model, batch, answer_mask, pad_mask=None, prompt_len=None, answer_window_len=None):
        batch = batch.long()
        if pad_mask is None:
            pad_mask = torch.ones_like(answer_mask, dtype=torch.bool)

        loss_mask = answer_mask.bool() & pad_mask.bool()
        if not torch.any(loss_mask):
            raise ValueError("all-mask CE batch has no answer tokens")

        mask_input = loss_mask
        if prompt_len is not None:
            positions = torch.arange(batch.shape[1], device=batch.device).unsqueeze(0)
            start = prompt_len.to(batch.device).long().unsqueeze(1)
            mask_input = positions >= start
            if answer_window_len is not None:
                mask_input = mask_input & (positions < start + int(answer_window_len))

        vocab_dim = graph.dim - 1 if graph.absorb else graph.dim

        t_idx = torch.randint(len(t_values), (), device=batch.device).item()
        ce_loss_mask = loss_mask
        if include_window_eos:
            # Keep the full fixed answer window masked as input, but do not train
            # CE on EOS-valued padding after the real answer boundary.
            ce_loss_mask = mask_input & pad_mask.bool()
        answer_weight_value = float(answer_weight)

        if cropped:
            if prompt_len is None or answer_window_len is None:
                raise ValueError("cropped all-mask CE requires prompt_len and answer_window_len")
            vals = []
            for row_idx in range(batch.shape[0]):
                start = int(prompt_len[row_idx].item())
                end = min(start + int(answer_window_len), batch.shape[1])
                if end <= start:
                    continue
                row = batch[row_idx : row_idx + 1, :end]
                row_loss_mask = ce_loss_mask[row_idx : row_idx + 1, :end]
                if not torch.any(row_loss_mask):
                    continue
                row_answer_mask = loss_mask[row_idx : row_idx + 1, :end]
                row_positions = torch.arange(end, device=batch.device).unsqueeze(0)
                row_mask_input = row_positions >= start
                masked_row = torch.where(
                    row_mask_input,
                    torch.full_like(row, graph.dim - 1),
                    row,
                )
                t = torch.full((1,), t_values[t_idx], device=batch.device)
                sigma, _ = noise(t)
                logits = model(masked_row, sigma)[..., :vocab_dim]
                token_loss = F.cross_entropy(
                    logits[row_loss_mask].float(),
                    row[row_loss_mask],
                    reduction="none",
                )
                weights = torch.ones_like(token_loss)
                if answer_weight_value != 1.0:
                    weights = torch.where(
                        row_answer_mask[row_loss_mask],
                        torch.full_like(weights, answer_weight_value),
                        weights,
                    )
                vals.append((token_loss * weights).sum() / weights.sum().clamp_min(1.0))
            if not vals:
                raise ValueError("cropped all-mask CE batch has no answer tokens")
            return torch.stack(vals).mean()

        masked_batch = torch.where(
            mask_input,
            torch.full_like(batch, graph.dim - 1),
            batch,
        )
        targets = batch[ce_loss_mask]
        t = torch.full((batch.shape[0],), t_values[t_idx], device=batch.device)
        sigma, _ = noise(t)
        logits = model(masked_batch, sigma)[..., :vocab_dim]
        token_loss = F.cross_entropy(logits[ce_loss_mask].float(), targets, reduction="none")
        weights = torch.ones_like(token_loss)
        if answer_weight_value != 1.0:
            weights = torch.where(
                loss_mask[ce_loss_mask],
                torch.full_like(weights, answer_weight_value),
                weights,
            )
        return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)

    return loss_fn


def _condition_len_to_answer_mask(batch, condition_len):
    positions = torch.arange(batch.shape[1], device=batch.device).unsqueeze(0)
    return positions >= condition_len.to(batch.device).long().unsqueeze(1)


def get_sft_loss_fn(noise, graph, train=True, sampling_eps=1e-3):
    """Backward-compatible SFT objective.

    New code should pass ``answer_mask`` and optional ``pad_mask``. Older
    diagnostics passed a 1-D ``condition_len`` tensor; that form is converted
    to an answer mask so those scripts still exercise the response-only DWDSE
    path instead of crashing.
    """
    response_loss_fn = get_response_only_sft_loss_fn(noise, graph, sampling_eps=sampling_eps)

    def loss_fn(model, batch, answer_mask_or_condition_len, pad_mask=None, sigma_max=None, **_):
        if answer_mask_or_condition_len.dim() == 1:
            answer_mask = _condition_len_to_answer_mask(batch, answer_mask_or_condition_len)
            if pad_mask is None:
                pad_mask = torch.ones_like(answer_mask, dtype=torch.bool)
        else:
            answer_mask = answer_mask_or_condition_len

        return response_loss_fn(
            model,
            batch,
            answer_mask,
            pad_mask=pad_mask,
            sigma_max=sigma_max,
        )

    return loss_fn
