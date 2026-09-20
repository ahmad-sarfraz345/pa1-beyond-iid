"""Generate and visually screen AdaIN cue conflicts without model predictions.

Uses the public naoto0804/pytorch-AdaIN implementation and its released weights.
The repository and weights must be provided locally; see task1/README.md.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T

from task1.data.make_subset import image224, load_dataset


def _stylizer(repo: Path, vgg_weights: Path, decoder_weights: Path, device):
    if not (repo / "net.py").exists() or not (repo / "function.py").exists():
        raise FileNotFoundError("AdaIN repo must contain net.py and function.py")
    sys.path.insert(0, str(repo.resolve()))
    import net
    from function import adaptive_instance_normalization

    vgg = net.vgg
    decoder = net.decoder
    vgg.load_state_dict(torch.load(vgg_weights, map_location="cpu", weights_only=True))
    decoder.load_state_dict(torch.load(decoder_weights, map_location="cpu", weights_only=True))
    return vgg[:31].eval().to(device), decoder.eval().to(device), adaptive_instance_normalization


def _save_sheet(rows, image_dir: Path, sheet_path: Path):
    # Show the content, style and generated image together for model-blind review.
    width, cell_h, columns = 3 * 224, 252, 4
    for page_start in range(0, len(rows), 20):
        page = rows[page_start : page_start + 20]
        sheet = Image.new("RGB", (width * columns, cell_h * 5), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, row in enumerate(page):
            x, y = (slot % columns) * width, (slot // columns) * cell_h
            for col, key in enumerate(("content_file", "style_file", "candidate_file")):
                with Image.open(image_dir / row[key]) as im:
                    sheet.paste(im, (x + col * 224, y + 25))
            draw.text((x + 3, y + 3), f"{row['id']} {row['content_class']} -> {row['style_class']}", fill="black")
        sheet.save(sheet_path.with_name(f"{sheet_path.stem}_{page_start//20:02d}.jpg"), quality=90)


def generate(data_root: Path, results: Path, adain_repo: Path, vgg_weights: Path,
             decoder_weights: Path, config: dict, device):
    manifest = json.loads((results / "splits.json").read_text())
    dataset = load_dataset(data_root, "test")
    image_dir = results / "cue_candidates"
    image_dir.mkdir(parents=True, exist_ok=True)
    encoder, decoder, adain = _stylizer(adain_repo, vgg_weights, decoder_weights, device)
    seed = config["seed"]
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.labels)
    class_ids = {name: i for i, name in enumerate(manifest["classes"])}
    selected = np.asarray(manifest["test_ids"])
    rows = []
    to_tensor = T.ToTensor()
    to_image = T.ToPILImage()
    alpha = config["adain_alpha"]
    for pair_id, (a, b) in enumerate(config["cue_pairs"]):
        for direction, (content_name, style_name) in enumerate(((a, b), (b, a))):
            content_pool = rng.permutation(selected[labels[selected] == class_ids[content_name]])
            style_pool = rng.permutation(selected[labels[selected] == class_ids[style_name]])
            n = min(config["cue_candidates_per_direction"], len(content_pool), len(style_pool))
            for j in range(n):
                cid, sid = int(content_pool[j]), int(style_pool[j])
                content, style = image224(dataset, cid), image224(dataset, sid)
                with torch.inference_mode():
                    cf = encoder(to_tensor(content).unsqueeze(0).to(device))
                    sf = encoder(to_tensor(style).unsqueeze(0).to(device))
                    mixed = alpha * adain(cf, sf) + (1 - alpha) * cf
                    generated = to_image(decoder(mixed).clamp(0, 1).squeeze(0).cpu())
                uid = f"p{pair_id}_d{direction}_{j:02d}"
                names = {key: f"{uid}_{key}.png" for key in ("content", "style", "candidate")}
                for key, im in (("content", content), ("style", style), ("candidate", generated)):
                    im.save(image_dir / names[key])
                rows.append({
                    "id": uid, "pair": pair_id, "direction": direction,
                    "content_class": content_name, "style_class": style_name,
                    "content_id": cid, "style_id": sid,
                    **{f"{k}_file": v for k, v in names.items()},
                    "accept": "", "reason": "",
                })
    fieldnames = list(rows[0])
    with (results / "cue_review.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _save_sheet(rows, image_dir, results / "cue_review.jpg")
    (results / "cue_rule.txt").write_text(config["cue_rejection_rule"] + "\n")
    return len(rows)


def accepted(results: Path, minimum: int = 200):
    with (results / "cue_review.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    pending = [r["id"] for r in rows if r["accept"].strip().lower() not in ("yes", "no")]
    if pending:
        raise ValueError(f"Review all cue candidates first; {len(pending)} pending")
    kept = [r for r in rows if r["accept"].strip().lower() == "yes"]
    counts = {}
    for row in rows:
        key = f"{row['pair']}_{row['direction']}"
        counts.setdefault(key, {"accepted": 0, "rejected": 0})
        counts[key]["accepted" if row in kept else "rejected"] += 1
    (results / "cue_review_counts.json").write_text(json.dumps(counts, indent=2))
    # Equal quotas prevent one easy-to-stylize pair from dominating shape bias.
    groups = sorted(counts)
    quota = max(1, (minimum + len(groups) - 1) // len(groups))
    if any(counts[group]["accepted"] < quota for group in groups):
        raise ValueError(f"Need {quota} accepted conflicts in every pair/direction; counts: {counts}")
    balanced = []
    for group in groups:
        balanced.extend([r for r in kept if f"{r['pair']}_{r['direction']}" == group][:quota])
    return balanced
