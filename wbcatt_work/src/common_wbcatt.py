"""
Model, patch-scoring rules, transforms and WBCATT data loading.

    build_model        ViT-B/16 (DenseViT) with stock ImageNet or LAST-ViT weights.
    cosine_scores      per-patch cosine similarity to the CLS token.
    fft_scores         per-patch LAST/FFT stability score.
    load_wbcatt_split  read an annotation CSV -> [(image_path, label), ...].

Paths resolved relative to this file:
    LAST-ViT/
    ├── last_vit_original/visualization/        DenseViT
    ├── dataset/data/wbcatt/
    │     ├── images/<celltype>/<img>.jpg
    │     └── annotations/pbc_attr_v1_{train,val,test}.csv
    ├── weights/ViT_190k.pth
    └── wbcatt_work/
"""
import csv
import sys
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# --------------------------- paths ---------------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]                       # src -> wbcatt_work -> LAST-ViT
PROJECT_DIR = _THIS.parents[1]                        # wbcatt_work/
ORIGINAL_REPO = PROJECT_ROOT / "last_vit_original"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUTS = PROJECT_DIR / "outputs"

WBCATT = PROJECT_ROOT / "dataset" / "data" / "wbcatt"
WBCATT_IMAGES = WBCATT / "images"
WBCATT_ANNOT = WBCATT / "annotations"

# DenseViT lives in the upstream repo's visualization/ dir
sys.path.insert(0, str(ORIGINAL_REPO / "visualization"))
from patch_score import DenseViT  # noqa: E402

# ----------------------- image geometry ----------------------
RESIZE, CROP, PATCH = 256, 224, 16
GRID = CROP // PATCH                                   # 14 -> 196 patches
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

EVAL_TF = transforms.Compose([
    transforms.Resize(RESIZE), transforms.CenterCrop(CROP),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
SHOW_TF = transforms.Compose([transforms.Resize(RESIZE), transforms.CenterCrop(CROP)])


# --------------------------- model ---------------------------
def build_model(device, checkpoint=None):
    """DenseViT with either the released LAST-ViT weights or stock torchvision weights."""
    model = DenseViT(image_size=CROP, patch_size=PATCH, num_layers=12,
                     num_heads=12, hidden_dim=768, mlp_dim=3072)
    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=False)
        print(f"loaded checkpoint {checkpoint} | missing={len(msg.missing_keys)} "
              f"unexpected={len(msg.unexpected_keys)}")
    else:
        ref = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.load_state_dict(ref.state_dict(), strict=False)
        print("loaded stock torchvision ImageNet ViT-B/16 weights")
    return model.to(device).eval()


# ----------------------- scoring rules -----------------------
def cosine_scores(cls_token, patch_tokens):
    """[B,N] cosine similarity of each patch token to the CLS token (baseline)."""
    return torch.cosine_similarity(patch_tokens, cls_token.unsqueeze(1), dim=-1)


def gaussian_kernel_1d(length, sigma, device, dtype=torch.float32):
    """Peak-normalised 1D Gaussian of size `length`, used to low-pass the feature axis."""
    positions = torch.arange(-length // 2 + 1, length // 2 + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    return kernel / kernel.max()


def fft_scores(patch_tokens, sigma=None, eps=1e-6):
    """
    [B,N] per-patch LAST stability score:
      - low-pass each token along the feature axis (FFT -> Gaussian -> iFFT)
      - stability = patch / |low_pass - patch|
      - per channel, topk(k=1) over patches picks the most-stable patch (the
        exact original-repo op); score[n] = #channels that picked patch n.
    sigma defaults to sqrt(hidden_dim) = sqrt(768) ~= 27.7.

    NOTE: the original repo returns a synthesized cls_token (gather + mean over
    the topk patches) for classification; it never emits a per-patch score.
    For visualization we need [B,N], so we histogram the same topk indices.
    """
    B, N, D = patch_tokens.shape
    if sigma is None:
        sigma = D ** 0.5
    X = torch.fft.fft(patch_tokens, dim=-1)
    X = torch.fft.fftshift(X, dim=-1)
    X = X * gaussian_kernel_1d(D, sigma, patch_tokens.device, patch_tokens.dtype)
    X = torch.fft.ifftshift(X, dim=-1)
    smooth = torch.fft.ifft(X, dim=-1).real
    diff = patch_tokens / (smooth - patch_tokens).abs().clamp_min(eps)   # [B,N,D] stability
    _, indices = torch.topk(diff, k=1, dim=1, largest=True)             # [B,1,D] original op
    winners = indices.squeeze(1)                                        # [B,D] per-channel winner
    scores = torch.zeros(B, N, device=patch_tokens.device)
    scores.scatter_add_(1, winners, torch.ones_like(winners, dtype=scores.dtype))
    return scores                                                       # [B,N]


# --------------------------- WBCATT data ---------------------
def _resolve_image_path(row):
    """
    On-disk path for a CSV row. CSV 'path' is 'PBC_dataset_normal_DIB/<celltype>/<img>.jpg';
    rebuild as images/<celltype>/<img_name>, falling back to a basename search.
    """
    celltype = Path(row["path"]).parent.name
    p = WBCATT_IMAGES / celltype / row["img_name"]
    if p.exists():
        return p
    hits = list(WBCATT_IMAGES.glob(f"*/{row['img_name']}"))
    return hits[0] if hits else p


def load_wbcatt_split(split="test", label=None):
    """
    [(abs_image_path, label), ...] for a split.

    split : 'train' | 'val' | 'test'
    label : optional case-insensitive class filter (e.g. 'Neutrophil'); None = all.
    """
    csv_path = WBCATT_ANNOT / f"pbc_attr_v1_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no annotation CSV at {csv_path}")
    want = label.lower() if label else None
    items = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if want and row["label"].lower() != want:
                continue
            items.append((str(_resolve_image_path(row)), row["label"]))
    return items
