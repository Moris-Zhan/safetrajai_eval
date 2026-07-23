# ============================================================
# reid_model.py  ─ 訓練工具函式
#
# 與舊版的差異：
#   - 不再使用 YOLOEmbeddingModel wrapper
#   - 以 ReIDHookManager 直接操作 ultralytics YOLO 原生模型
#   - 儲存時呼叫 save_yolo_reid()，維持 ultralytics 原生格式
# ============================================================
import copy, os, json, time
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from ultralytics import YOLO

from module.reid.yolo_reid_trainer import (
    ReIDHookManager,
    BatchHardTripletLoss,
    save_yolo_reid,
)


# ──────────────────────────────────────────────────────────────
# 1. Early Stopping
# ──────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.counter    = 0
        self.best_loss  = float("inf")
        self.early_stop = False
        # 分別儲存 backbone 與 projector 的最佳狀態
        self.best_backbone_state   = None
        self.best_projector_state  = None

    def __call__(self, val_loss: float, yolo: YOLO, manager: ReIDHookManager):
        if self.best_loss - val_loss > self.min_delta:
            if self.verbose:
                print(f"  ↓ val_loss {self.best_loss:.6f} → {val_loss:.6f}  (saved)")
            self.best_loss = val_loss
            self.best_backbone_state  = copy.deepcopy(yolo.model.state_dict())
            self.best_projector_state = copy.deepcopy(manager.projector.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best(self, yolo: YOLO, manager: ReIDHookManager):
        """將 backbone 與 projector 還原為最佳權重。"""
        if self.best_backbone_state:
            yolo.model.load_state_dict(self.best_backbone_state)
        if self.best_projector_state:
            manager.projector.load_state_dict(self.best_projector_state)
            # 同步更新附掛在 yolo.model 上的 projector
            yolo.model.reid_projector = manager.projector


# ──────────────────────────────────────────────────────────────
# 2. 工具函式
# ──────────────────────────────────────────────────────────────
def get_next_versioned_folder(base_dir: str = "runs/triplet-yolo",
                              prefix: str = "train-") -> str:
    os.makedirs(base_dir, exist_ok=True)
    nums = [
        int(d.replace(prefix, ""))
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and d.startswith(prefix)
        and d.replace(prefix, "").isdigit()
    ]
    version = max(nums) + 1 if nums else 1
    folder  = os.path.join(base_dir, f"{prefix}{version:03d}")
    os.makedirs(folder, exist_ok=True)
    return folder


def build_transform(params: dict) -> transforms.Compose:
    aug = params["augmentation"]
    t   = [transforms.Resize((params["img_size"], params["img_size"]))]

    if aug["enabled"]:
        if aug["random_horizontal_flip"] > 0:
            t.append(transforms.RandomHorizontalFlip(p=aug["random_horizontal_flip"]))
        if aug["random_rotation"] > 0:
            t.append(transforms.RandomRotation(degrees=aug["random_rotation"]))
        if aug["color_jitter"]["enabled"]:
            cj = aug["color_jitter"]
            t.append(transforms.ColorJitter(
                brightness=cj["brightness"], contrast=cj["contrast"],
                saturation=cj["saturation"], hue=cj["hue"]))
        if aug["random_affine"]["enabled"]:
            ra = aug["random_affine"]
            t.append(transforms.RandomAffine(
                degrees=0, translate=ra["translate"], scale=ra["scale"]))

    t.append(transforms.ToTensor())

    if aug["normalize"]["enabled"]:
        n = aug["normalize"]
        t.append(transforms.Normalize(mean=n["mean"], std=n["std"]))

    if aug["enabled"] and aug["random_erasing"]["enabled"]:
        re = aug["random_erasing"]
        t.append(transforms.RandomErasing(p=re["p"], scale=re["scale"], ratio=re["ratio"]))

    return transforms.Compose(t)


# ──────────────────────────────────────────────────────────────
# 3. 訓練主流程
# ──────────────────────────────────────────────────────────────
def train(train_params: dict):
    import wandb
    from tqdm import tqdm

    device    = train_params["device"]
    transform = build_transform(train_params)

    train_dataset = datasets.ImageFolder(train_params["data_path"],     transform=transform)
    val_dataset   = datasets.ImageFolder(train_params["data_path_val"], transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=train_params["batch_size"],
                               shuffle=train_params["shuffle"])
    val_loader    = DataLoader(val_dataset,   batch_size=train_params["batch_size"],
                               shuffle=False)

    print(f"Classes: {len(train_dataset.classes)}  "
          f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # ── 載入 ultralytics YOLO 原生物件 ──────────────────────
    yolo = YOLO(train_params["yolo_weights"])
    yolo.model.to(device)

    # ── 建立 hook manager（同時建立並附掛 projector）──────────
    manager = ReIDHookManager(yolo, device=device)

    # ── 凍結 backbone（若啟用）──────────────────────────────
    manager.freeze_backbone(train_params.get("freeze_backbone", False))

    # ── Loss / Optimizer / Scheduler ────────────────────────
    criterion = BatchHardTripletLoss(margin=train_params.get("margin", 0.3)).to(device)

    projector_ids = {id(p) for p in manager.projector.parameters()}
    backbone_params = [p for p in yolo.model.parameters()
                       if id(p) not in projector_ids]

    optimizer = torch.optim.Adam([
        {"params": backbone_params,                "lr": train_params["lr"] * 0.1},
        {"params": manager.projector.parameters(), "lr": train_params["lr"]},
    ], weight_decay=train_params.get("weight_decay", 1e-4))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_params["epochs"], eta_min=train_params["lr"] * 0.01)

    es_cfg = train_params["early_stopping"]
    early_stopping = (
        EarlyStopping(
            patience=es_cfg["patience"],
            min_delta=es_cfg["min_delta"],
            verbose=es_cfg["verbose"],
        )
        if es_cfg["enabled"] else None
    )

    wandb.init(project="reid-training", name=train_params["project"], config=train_params)
    wandb.watch(yolo.model, log="all", log_freq=100)

    print(f"\n{'Epoch':<8}{'GPU_mem':<12}{'train/loss':<15}{'val/loss':<15}{'lr':<12}")
    print("-" * 60)

    for epoch in range(train_params["epochs"]):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────
        manager.train_mode()
        total_loss, valid_batches = 0.0, 0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{train_params['epochs']}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            embed = manager.embed(imgs)
            loss  = criterion(embed, labels)
            if loss.item() == 0:
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(yolo.model.parameters()) + list(manager.projector.parameters()),
                max_norm=1.0)
            optimizer.step()
            total_loss += loss.item(); valid_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train = total_loss / max(valid_batches, 1)

        # ── Validation ───────────────────────────────────────
        manager.eval_mode()
        total_val, valid_val = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                embed = manager.embed(imgs)
                loss  = criterion(embed, labels)
                if loss.item() > 0:
                    total_val += loss.item(); valid_val += 1
        avg_val = total_val / max(valid_val, 1)

        scheduler.step()
        lr  = optimizer.param_groups[-1]["lr"]
        mem = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
        print(f"{epoch+1:<8}{mem:.2f}G{'':<5}"
              f"{avg_train:.5f}{'':<8}{avg_val:.5f}{'':<3}{lr:.2e}")

        wandb.log({
            "epoch":            epoch + 1,
            "train/loss":       avg_train,
            "val/loss":         avg_val,
            "lr":               lr,
            "gpu_mem(GB)":      mem,
            "epoch_time(sec)":  time.time() - t0,
            "valid_batches":    valid_batches,
        })

        if early_stopping:
            early_stopping(avg_val, yolo, manager)
            if early_stopping.early_stop:
                print("\n⚠️  Early stopping triggered")
                early_stopping.restore_best(yolo, manager)
                break

    # ── 儲存：ultralytics 原生格式 ────────────────────────────
    save_path = os.path.join(train_params["project"], "best.pt")
    save_yolo_reid(yolo, save_path, train_args=train_params)

    manager.remove_hook()
    wandb.finish()
    return yolo