from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Usage
# pixi run python query_color.py "bag of cookies" --dataset_name teatime --output query_color_bag_of_cookies.jpg

def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def encode_query_text(query: str, device):
    import open_clip
    import torch

    precision = "fp16" if device.type == "cuda" else "fp32"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        precision=precision,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        text_feature = model.encode_text(tokens).float()
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

    return text_feature


def load_autoencoder(args: argparse.Namespace, device):
    import torch
    from autoencoder.model import Autoencoder

    ckpt_path = args.ae_ckpt
    if ckpt_path is None:
        ckpt_path = BASE_DIR / "autoencoder" / "ckpt" / args.dataset_name / "best_ckpt.pth"
    ckpt_path = ckpt_path.expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ckpt_path}")

    if args.print_model:
        model = Autoencoder(args.encoder_dims, args.decoder_dims).to(device)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            model = Autoencoder(args.encoder_dims, args.decoder_dims).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def transform_color_feature(color_feature, transform: str):
    import torch

    color_feature = color_feature / (color_feature.norm(dim=-1, keepdim=True) + 1e-9)
    if transform == "render":
        return color_feature
    if transform == "signed":
        return (color_feature + 1.0) * 0.5
    if transform == "sigmoid":
        return torch.sigmoid(color_feature)
    raise ValueError(f"Unknown color transform: {transform}")


def render_color_square(color_feature, output_path: Path, size: int, transform: str) -> None:
    import torchvision

    if color_feature.shape != (3,):
        raise ValueError(f"Expected a 3D encoded feature, got shape {tuple(color_feature.shape)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_feature = transform_color_feature(color_feature, transform)
    square = display_feature.view(3, 1, 1).repeat(1, size, size).cpu()
    torchvision.utils.save_image(square, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a natural-language query to LangSplat 3D feature color and save a filled JPG."
    )
    parser.add_argument("query", help="Natural-language query, e.g. 'coffee mug'.")
    parser.add_argument(
        "--dataset_name",
        default="teatime",
        help="Dataset name under autoencoder/ckpt/. Used when --ae_ckpt is not set.",
    )
    parser.add_argument(
        "--ae_ckpt",
        type=Path,
        default=None,
        help="Path to autoencoder best_ckpt.pth. Overrides --dataset_name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("query_color.jpg"),
        help="Output JPG path.",
    )
    parser.add_argument("--size", type=int, default=256, help="Output square size in pixels.")
    parser.add_argument(
        "--color_transform",
        choices=["render", "signed", "sigmoid"],
        default="render",
        help=(
            "How to map the 3D feature to an image. "
            "'render' matches render.py/save_image clamping; 'signed' maps [-1,1] to [0,1]."
        ),
    )
    parser.add_argument(
        "--encoder_dims",
        nargs="+",
        type=int,
        default=[256, 128, 64, 32, 3],
    )
    parser.add_argument(
        "--decoder_dims",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 256, 512],
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Defaults to auto.",
    )
    parser.add_argument(
        "--print_model",
        action="store_true",
        help="Print the autoencoder architecture emitted by Autoencoder.__init__.",
    )
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    return args


def main() -> None:
    args = parse_args()
    import torch

    device = resolve_device(args.device)

    text_feature = encode_query_text(args.query, device)
    autoencoder = load_autoencoder(args, device)
    with torch.no_grad():
        color_feature = autoencoder.encode(text_feature).squeeze(0)

    display_feature = transform_color_feature(color_feature, args.color_transform)
    render_color_square(color_feature, args.output.expanduser(), args.size, args.color_transform)

    render_rgb = transform_color_feature(color_feature, "render").detach().cpu().clamp(0.0, 1.0).mul(255).round().int().tolist()
    display_rgb = display_feature.detach().cpu().clamp(0.0, 1.0).mul(255).round().int().tolist()
    print(f"query: {args.query}")
    print(f"encoded_3d: {color_feature.detach().cpu().tolist()}")
    print(f"render_rgb_0_255: {render_rgb}")
    print(f"saved_rgb_0_255: {display_rgb} ({args.color_transform})")
    print(f"saved: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
