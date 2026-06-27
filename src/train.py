# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────
from helper.cfg import CFG
from helper.makros import DEVICE

def compute_loss(outputs, labels):
    total = 0.0
    for task, weight in CFG["task_weights"].items():
        total += weight * LOSS_FNS[task](outputs[task], labels[task].to(DEVICE))
    return total

def run_epoch(loader, training=True):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    for imgs, ids, masks, labels in loader:
        imgs  = imgs.to(DEVICE)
        ids   = ids.to(DEVICE)
        masks = masks.to(DEVICE)

        with autocast():
            outputs = model(imgs, ids, masks)
            loss    = compute_loss(outputs, labels)

        if training:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        total_loss += loss.item()
        preds  = outputs["sentiment"].argmax(dim=1).cpu()
        truths = labels["sentiment"]
        correct += (preds == truths).sum().item()
        total   += len(truths)

    return total_loss / len(loader), correct / total


HISTORY = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0
patience_cnt = 0
best_ckpt    = os.path.join(CFG["out_dir"], "best_memotion_v2.pt")

print("─" * 60)
print(f"{'Epoch':>5} {'T-Loss':>8} {'T-Acc':>7} {'V-Loss':>8} {'V-Acc':>7}")
print("─" * 60)

for epoch in range(1, CFG["epochs"] + 1):
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    with torch.no_grad():
        vl_loss, vl_acc = run_epoch(val_loader, training=False)

    HISTORY["train_loss"].append(tr_loss)
    HISTORY["val_loss"].append(vl_loss)
    HISTORY["train_acc"].append(tr_acc)
    HISTORY["val_acc"].append(vl_acc)

    print(f"{epoch:5d} {tr_loss:8.4f} {tr_acc:7.4f} {vl_loss:8.4f} {vl_acc:7.4f}")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        patience_cnt = 0
        torch.save(model.state_dict(), best_ckpt)
        print(f"       ✅ New best val_acc = {best_val_acc:.4f} — checkpoint saved")
    else:
        patience_cnt += 1
        if patience_cnt >= CFG["patience"]:
            print(f"⏹  Early stopping at epoch {epoch}")
            break

print("─" * 60)
print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")