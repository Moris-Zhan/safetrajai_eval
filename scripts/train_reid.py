"""
YOLO-as-encoder Re-ID 訓練腳本（單一類別一次）。

資料來源：track_eval.py 的 save_reid_crops 蒐集的 dataset/reid/crop_dataset/{class}/{id}/*.jpg。
模型架構：safetrajai/reid/yolo_reid_trainer.py 的 ReIDHookManager（YOLO 倒數第二層 + projector），
         不使用 utils/model.py 的 ReIDNet（那個架構需要 camera ID，訓練程式碼也不在本專案內）。
訓練出的權重可直接被 safetrajai/reid/encoders.py::ReID 載入（走 safetrajai/tracking/class_aware_botsort.py
的「非 reid_model_weights、非 class_reid_weights」分支）。

用法
----
python scripts/train_reid.py --class-name person \
    --yolo-weights models/yolo/yolo11x_dennis.pt \
    --output models/yolo-reid/person_reid.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from safetrajai.reid.training.dataset import (
    ReIDCropDataset,
    PKSampler,
    discover_identities,
    split_identities,
)
from safetrajai.reid.yolo_reid_trainer import (
    ReIDHookManager,
    BatchHardTripletLoss,
    save_yolo_reid,
)


class EarlyStopping:
    """依 val loss 判斷是否早停，並記錄 backbone/projector 的最佳狀態。"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_backbone_state = None
        self.best_projector_state = None

    def __call__(self, val_loss: float, yolo, manager: ReIDHookManager):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_backbone_state = {
                k: v.detach().cpu().clone() for k, v in yolo.model.state_dict().items()
            }
            self.best_projector_state = {
                k: v.detach().cpu().clone() for k, v in manager.projector.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best(self, yolo, manager: ReIDHookManager):
        if self.best_backbone_state is not None:
            yolo.model.load_state_dict(self.best_backbone_state)
        if self.best_projector_state is not None:
            manager.projector.load_state_dict(self.best_projector_state)


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_epoch(manager: ReIDHookManager, loader: DataLoader, criterion, optimizer, device, train: bool) -> float:
    manager.train_mode() if train else manager.eval_mode()
    total_loss, n_batches = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            embed = manager.embed(imgs)
            loss = criterion(embed, labels)

        if train and loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def train(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    device = args.device
    identity_dirs = discover_identities(args.data_root, args.class_name)
    train_dirs, val_dirs = split_identities(identity_dirs, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[{args.class_name}] 身分數：train={len(train_dirs)}  val={len(val_dirs)}")

    transform = build_transform(args.img_size)
    train_dataset = ReIDCropDataset(train_dirs, transform=transform, min_images_per_identity=args.min_images)
    val_dataset = ReIDCropDataset(val_dirs, transform=transform, min_images_per_identity=args.min_images)
    print(f"[{args.class_name}] 有效身分數（影像數足夠）：train={train_dataset.num_identities}  val={val_dataset.num_identities}")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PKSampler(train_dataset.labels, P=args.P, K=args.K),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=PKSampler(val_dataset.labels, P=min(args.P, val_dataset.num_identities), K=args.K),
    )

    yolo = YOLO(args.yolo_weights)
    yolo.model.to(device)
    manager = ReIDHookManager(yolo, device=device)
    manager.freeze_backbone(args.freeze_backbone)

    criterion = BatchHardTripletLoss(margin=args.margin).to(device)

    projector_ids = {id(p) for p in manager.projector.parameters()}
    backbone_params = [p for p in yolo.model.parameters() if id(p) not in projector_ids]
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": manager.projector.parameters(), "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    early_stopping = EarlyStopping(patience=args.patience) if args.patience > 0 else None

    if args.wandb:
        import wandb
        wandb.init(project="reid-training", name=f"{args.class_name}-{args.yolo_weights}", config=vars(args))

    print(f"\n{'Epoch':<8}{'train/loss':<15}{'val/loss':<15}{'lr':<12}")
    print("-" * 50)
    for epoch in range(args.epochs):
        train_loss = run_epoch(manager, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(manager, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        lr_now = optimizer.param_groups[-1]["lr"]
        print(f"{epoch + 1:<8}{train_loss:<15.4f}{val_loss:<15.4f}{lr_now:<12.6f}")

        if args.wandb:
            import wandb
            wandb.log({"epoch": epoch + 1, "train/loss": train_loss, "val/loss": val_loss, "lr": lr_now})

        if early_stopping is not None:
            early_stopping(val_loss, yolo, manager)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1} (best val loss = {early_stopping.best_loss:.4f})")
                break

    if early_stopping is not None and early_stopping.best_backbone_state is not None:
        early_stopping.restore_best(yolo, manager)

    manager.remove_hook()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_yolo_reid(yolo, str(output_path), train_args=vars(args))
    return output_path


def main():
    parser = argparse.ArgumentParser(description="YOLO-as-encoder Re-ID 訓練（單一類別）")
    parser.add_argument("--class-name", required=True, help="要訓練的類別（對應 dataset/reid/crop_dataset/{class}/）")
    parser.add_argument("--data-root", default="dataset/reid/crop_dataset", help="Re-ID 訓練資料根目錄")
    parser.add_argument("--yolo-weights", default="models/yolo/yolo11x_dennis.pt", help="做為 encoder 起點的 YOLO 權重")
    parser.add_argument("--output", default=None, help="輸出權重路徑，預設 models/yolo-reid/{class}_reid.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--P", type=int, default=8, help="每個 batch 抽幾個身分")
    parser.add_argument("--K", type=int, default=4, help="每個身分抽幾張影像")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="依身分切分的驗證集比例")
    parser.add_argument("--min-images", type=int, default=4, help="身分資料夾影像數低於此值就捨棄該身分")
    parser.add_argument("--freeze-backbone", action="store_true", help="只訓練 projector，不動 YOLO backbone")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping 耐心值，<=0 關閉")
    parser.add_argument("--seed", type=int, default=42, help="train/val 身分切分的隨機種子")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="是否用 wandb 記錄訓練過程（預設關閉）")

    args = parser.parse_args()
    if args.output is None:
        args.output = f"models/yolo-reid/{args.class_name}_reid.pt"

    train(args)


if __name__ == "__main__":
    main()
