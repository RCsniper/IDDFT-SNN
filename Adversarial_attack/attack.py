import json
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

import torch


try:
    from snntorch import utils as snn_utils  # optional
except Exception:
    snn_utils = None


# ============================================================
# Basic helpers
# ============================================================
def normalize_input(x: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def reset_snn_state(model: torch.nn.Module) -> None:
    """Best-effort reset for stateful SNN modules."""
    if snn_utils is not None and hasattr(snn_utils, "reset"):
        try:
            snn_utils.reset(model)
            return
        except Exception:
            pass

    called = set()
    for module in model.modules():
        for fn_name in (
            "reset",
            "reset_state",
            "reset_states",
            "clear_state",
            "reset_mem",
            "init_leaky",):
            fn = getattr(module, fn_name, None)
            if callable(fn):
                fn_id = (id(module), fn_name)
                if fn_id not in called:
                    fn()
                    called.add(fn_id)


def _reduce_logits(logits_t: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Convert temporal logits to [B, C].
    Supports [B, T, C] or [T, B, C].
    """
    if logits_t.ndim == 3:
        if logits_t.shape[0] == batch_size:
            return logits_t.mean(dim=1)
        if logits_t.shape[1] == batch_size:
            return logits_t.mean(dim=0)
    return logits_t


def forward_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    mean,
    std,
    reset_state: bool = True,
) -> torch.Tensor:
    if reset_state:
        reset_snn_state(model)

    images_norm = normalize_input(images, mean, std)
    logits_t = model(images_norm)
    logits = _reduce_logits(logits_t, batch_size=images.shape[0])
    return logits


@torch.no_grad()
def forward_and_predict(
    model: torch.nn.Module,
    images: torch.Tensor,
    mean,
    std,
    reset_state: bool = True,
) -> torch.Tensor:
    logits = forward_logits(model, images, mean, std, reset_state=reset_state)
    return logits.argmax(dim=1)


@torch.no_grad()
def evaluate_clean(
    model: torch.nn.Module,
    loader,
    mean,
    std,
    device,
    reset_state: bool = True,
) -> float:
    model.eval()
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        pred = forward_and_predict(model, images, mean, std, reset_state=reset_state)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


# ============================================================
# Attack implementations (source model creates adversarial examples)
# ============================================================
def fgsm_attack(
    model_attack: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: Callable,
    eps: float,
    mean,
    std,
    device,
    reset_state: bool = True,
) -> torch.Tensor:
    model_attack.eval()

    images = images.detach().clone().to(device)
    labels = labels.detach().clone().to(device)
    images.requires_grad_(True)

    logits = forward_logits(model_attack, images, mean, std, reset_state=reset_state)
    loss = loss_fn(logits, labels)

    grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()
    return adv_images


def rfgsm_attack(
    model_attack: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: Callable,
    eps: float,
    alpha: float,
    mean,
    std,
    device,
    reset_state: bool = True,
) -> torch.Tensor:
    """
    Randomized FGSM:
      1) random start within [-alpha, alpha]
      2) one FGSM step using remaining budget (eps - alpha)
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative for RFGSM.")
    if alpha > eps:
        raise ValueError("For RFGSM, alpha should not exceed eps.")

    model_attack.eval()

    images = images.detach().clone().to(device)
    labels = labels.detach().clone().to(device)

    init_noise = torch.empty_like(images).uniform_(-alpha, alpha)
    adv_images = torch.clamp(images + init_noise, 0.0, 1.0).detach()
    adv_images.requires_grad_(True)

    logits = forward_logits(model_attack, adv_images, mean, std, reset_state=reset_state)
    loss = loss_fn(logits, labels)

    grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

    adv_images = adv_images.detach() + (eps - alpha) * grad.sign()
    adv_images = torch.max(torch.min(adv_images, images + eps), images - eps)
    adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()
    return adv_images


def pgd_attack(
    model_attack: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: Callable,
    eps: float,
    alpha: float,
    steps: int,
    mean,
    std,
    device,
    random_start: bool = True,
    reset_state: bool = True,
) -> torch.Tensor:
    model_attack.eval()

    images = images.detach().clone().to(device)
    labels = labels.detach().clone().to(device)

    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()
    else:
        adv_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)

        logits = forward_logits(model_attack, adv_images, mean, std, reset_state=reset_state)
        loss = loss_fn(logits, labels)

        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        adv_images = torch.max(torch.min(adv_images, images + eps), images - eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

    return adv_images


ATTACK_REGISTRY = {
    "fgsm": fgsm_attack,
    "rfgsm": rfgsm_attack,
    "pgd": pgd_attack,
}


# ============================================================
# Transfer evaluation
# ============================================================
def evaluate_transfer(
    target_model: torch.nn.Module,
    source_model: torch.nn.Module,
    loader,
    attack_name: str,
    attack_kwargs: Dict[str, Any],
    loss_fn: Callable,
    mean,
    std,
    device,
    reset_state: bool = True,
) -> Dict[str, Any]:
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack_name: {attack_name}")

    target_model.eval()
    source_model.eval()

    total = 0
    correct = 0
    total_clean_correct = 0
    robust_on_clean_correct = 0

    attack_fn = ATTACK_REGISTRY[attack_name]

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            clean_pred = forward_and_predict(
                target_model, images, mean, std, reset_state=reset_state
            )
            clean_correct_mask = clean_pred.eq(labels)

        adv_images = attack_fn(
            model_attack=source_model,
            images=images,
            labels=labels,
            loss_fn=loss_fn,
            mean=mean,
            std=std,
            device=device,
            reset_state=reset_state,
            **attack_kwargs,
        )
        with torch.no_grad():
            adv_pred = forward_and_predict(
                target_model, adv_images, mean, std, reset_state=reset_state
            )
        correct_mask = adv_pred.eq(labels)

        correct += correct_mask.sum().item()
        total += labels.size(0)
        total_clean_correct += clean_correct_mask.sum().item()
        robust_on_clean_correct += (correct_mask & clean_correct_mask).sum().item()

    adv_acc_full = correct / max(total, 1)
    robust_acc = robust_on_clean_correct / max(total_clean_correct, 1)

    return {
        "adv_acc_full": adv_acc_full,
        "robust_acc": robust_acc,
        "num_clean_correct": total_clean_correct,
        "num_total": total,
    }


def evaluate_transfer_matrix(
    models: OrderedDict,
    loader,
    attack_name: str,
    attack_kwargs: Dict[str, Any],
    loss_fn: Callable,
    mean,
    std,
    device,
    reset_state: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    names = list(models.keys())
    results = {src: {} for src in names}

    for src_name, src_model in models.items():
        for tgt_name, tgt_model in models.items():
            out = evaluate_transfer(
                target_model=tgt_model,
                source_model=src_model,
                loader=loader,
                attack_name=attack_name,
                attack_kwargs=attack_kwargs,
                loss_fn=loss_fn,
                mean=mean,
                std=std,
                device=device,
                reset_state=reset_state,
            )
            results[src_name][tgt_name] = out
            # print(
            #     f"source={src_name:>5} -> target={tgt_name:>5} | "
            #     f"adv_acc_full={out['adv_acc_full'] * 100:.2f}% | "
            #     f"robust_acc={out['robust_acc'] * 100:.2f}%"
            # )
    return results


# ============================================================
# Reporting helpers
# ============================================================
def extract_metric_matrix(results, metric_key: str) -> Dict[str, Dict[str, float]]:
    matrix = {src: {} for src in results.keys()}
    for src in results.keys():
        for tgt in results[src].keys():
            matrix[src][tgt] = results[src][tgt][metric_key]
    return matrix


def print_matrix(matrix: Dict[str, Dict[str, float]], title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    headers = list(matrix.keys())
    print("{:>15}".format("source\\target"), end="")
    for h in headers:
        print(f"{h:>15}", end="")
    print()

    for src in headers:
        print(f"{src:>15}", end="")
        for tgt in headers:
            print(f"{matrix[src][tgt] * 100:>14.2f}%", end="")
        print()
    print("=" * 72)


def save_results_json(
    save_path: str,
    clean_acc: Dict[str, float],
    attack_name: str,
    attack_kwargs: Dict[str, Any],
    raw_results: Dict[str, Dict[str, Dict[str, Any]]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "clean_acc": clean_acc,
        "attack_name": attack_name,
        "attack_kwargs": attack_kwargs,
        "raw_results": raw_results,
    }
    if extra is not None:
        payload.update(extra)

    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
