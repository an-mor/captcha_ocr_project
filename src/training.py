from contextlib import nullcontext
from copy import deepcopy

import torch

from .metrics import character_error_rate, decode_targets, edit_distance


def _autocast_context(device, use_amp):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=True):
    model.train()
    total_loss = 0.0

    for images, targets, _, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, criterion, device, idx_to_char, use_amp=True):
    model.eval()
    total_loss = 0.0
    references = []
    predictions = []
    records = []

    for images, targets, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        predicted_indices = logits.argmax(dim=-1).cpu()
        batch_predictions = decode_targets(predicted_indices, idx_to_char)
        references.extend(labels)
        predictions.extend(batch_predictions)
        total_loss += loss.item() * images.size(0)

        for reference, prediction, path in zip(labels, batch_predictions, paths):
            distance = edit_distance(reference, prediction)
            records.append(
                {
                    "path": path,
                    "reference": reference,
                    "prediction": prediction,
                    "distance": distance,
                    "sample_cer": distance / len(reference),
                }
            )

    return {
        "loss": total_loss / len(loader.dataset),
        "cer": character_error_rate(references, predictions),
        "exact_accuracy": sum(
            reference == prediction
            for reference, prediction in zip(references, predictions)
        )
        / len(references),
        "records": records,
    }


def fit(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    idx_to_char,
    epochs,
    early_stopping_patience=7,
    use_amp=True,
):
    history = []
    best_cer = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp,
        )
        metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            idx_to_char,
            use_amp,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": metrics["loss"],
                "validation_cer": metrics["cer"],
                "validation_exact_accuracy": metrics["exact_accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        scheduler.step(metrics["loss"])

        if metrics["cer"] < best_cer:
            best_cer = metrics["cer"]
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={metrics['loss']:.4f}, "
            f"val_CER={metrics['cer']:.4f}, "
            f"val_accuracy={metrics['exact_accuracy']:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping after {epoch} epochs; "
                f"best validation CER={best_cer:.4f}."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
