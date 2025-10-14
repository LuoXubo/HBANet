import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from models.network_hbanet import HBANet
from utils import utils_image as util
from data.dataloder import Dataset as FusionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HBANet infrared-visible image fusion inference")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained checkpoint (.pth)')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory containing test datasets')
    parser.add_argument('--dataset', type=str, default='MSRS', help='Dataset name inside dataset_root')
    parser.add_argument('--ir_dir', type=str, default='IR', help='Infrared sub-directory name')
    parser.add_argument('--vis_dir', type=str, default='VI', help='Visible sub-directory name')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save fused results')
    parser.add_argument('--in_channels', type=int, default=1, help='Input channels per modality (1 or 3)')
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of feature channels in encoder')
    parser.add_argument('--encoder_blocks', type=int, default=4, help='Number of residual blocks in encoder')
    parser.add_argument('--decoder_blocks', type=int, default=2, help='Number of residual blocks in decoder')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in CDAU')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda', help="'cuda' or 'cpu'")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> HBANet:
    model = HBANet(
        in_chans=args.in_channels,
        base_channels=args.base_channels,
        encoder_res_blocks=args.encoder_blocks,
        decoder_res_blocks=args.decoder_blocks,
        num_heads=args.num_heads,
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('params', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    ir_root = os.path.join(args.dataset_root, args.dataset, args.ir_dir)
    vis_root = os.path.join(args.dataset_root, args.dataset, args.vis_dir)
    if not os.path.isdir(ir_root) or not os.path.isdir(vis_root):
        raise FileNotFoundError(f"Input folders not found: {ir_root}, {vis_root}")
    dataset = FusionDataset(ir_root, vis_root, args.in_channels)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


def fuse_and_save(model: HBANet, dataloader: DataLoader, args: argparse.Namespace, device: torch.device) -> None:
    output_root = os.path.join(args.output_dir, f"HBANet_{args.dataset}")
    os.makedirs(output_root, exist_ok=True)

    for idx, data in enumerate(dataloader, start=1):
        img_ir = data['A'].to(device)
        img_vis = data['B'].to(device)
        name = os.path.basename(data['A_path'][0])

        start = time.time()
        with torch.no_grad():
            fused = model(img_ir, img_vis)
        elapsed = time.time() - start

        fused_uint = util.tensor2uint(fused[0].cpu())
        save_path = os.path.join(output_root, name)
        util.imsave(fused_uint, save_path)

        print(f"[{idx}/{len(dataloader)}] Saved {save_path} | {elapsed:.4f}s")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.model_path):
        print(f"Checkpoint not found: {args.model_path}")
        sys.exit(1)

    model = load_model(args, device)
    dataloader = build_dataloader(args)
    fuse_and_save(model, dataloader, args, device)


if __name__ == '__main__':
    main()
