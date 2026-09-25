"""Task 1 CLI. Run from repository root: python -m task1.scripts.run_task1 ..."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from task1.analysis.evaluate_bias import consistency, cue_counts, metrics
from task1.analysis.feature_similarity import cosine_stability
from task1.analysis.representation import plot_projection, plot_translation
from task1.data.make_cue_conflicts import accepted, generate
from task1.data.make_subset import image224, load_dataset, prepare
from task1.data.transforms import variants
from task1.models.backbones import FrozenBackbone


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "task1" / "results"
CONFIG = ROOT / "task1" / "configs" / "task1.json"
NAMES = ("resnet50", "vit_b_16", "clip_vit_b_32")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def batched_features(model, images, batch_size, label):
    chunks = []
    for start in tqdm(range(0, len(images), batch_size), desc=label, leave=False):
        chunks.append(model.encode(images[start : start + batch_size]).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_head(train_x, train_y, val_x, val_y, config, device):
    seed_all(config["seed"])
    head = nn.Linear(train_x.shape[1], len(np.unique(train_y))).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["head_lr"],
                                  weight_decay=config["head_weight_decay"])
    xtr = torch.from_numpy(train_x).float().to(device)
    ytr = torch.as_tensor(train_y, dtype=torch.long, device=device)
    xval = torch.from_numpy(val_x).float().to(device)
    yval = np.asarray(val_y)
    generator = torch.Generator().manual_seed(config["seed"])
    best_acc, best_epoch, best_weights = -1.0, 0, None
    history = []
    for epoch in range(1, config["head_max_epochs"] + 1):
        head.train()
        order = torch.randperm(len(xtr), generator=generator)
        for indices in order.split(config["batch_size"]):
            indices = indices.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(head(xtr[indices]), ytr[indices])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            val_prediction = head(xval).argmax(1).cpu().numpy()
        acc = float(np.mean(val_prediction == yval))
        history.append({"epoch": epoch, "validation_accuracy": acc})
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        if epoch - best_epoch >= config["head_patience"]:
            break
    head.load_state_dict(best_weights)
    return head.eval(), {"best_epoch": best_epoch, "best_validation_accuracy": best_acc,
                         "history": history}


def probability(head, features, device):
    with torch.inference_mode():
        logits = head(torch.from_numpy(features).float().to(device))
        return logits.softmax(dim=1).cpu().numpy()


def zero_shot_probability(features, texts, scale, device):
    with torch.inference_mode():
        x = F.normalize(torch.from_numpy(features).float().to(device), dim=-1)
        return (scale * x @ texts.T).softmax(dim=1).cpu().numpy()


def train_command(config, manifest, device, selected_model=None):
    dataset = load_dataset(DATA, "train")
    for name in (selected_model,) if selected_model else NAMES:
        model = FrozenBackbone(name, device)
        train_ids, val_ids = manifest["train_ids"], manifest["val_ids"]
        train_x = batched_features(model, [image224(dataset, i) for i in train_ids],
                                   config["batch_size"], name + " train")
        val_x = batched_features(model, [image224(dataset, i) for i in val_ids],
                                 config["batch_size"], name + " val")
        train_y = np.asarray(dataset.labels)[train_ids]
        val_y = np.asarray(dataset.labels)[val_ids]
        head, history = train_head(train_x, train_y, val_x, val_y, config, device)
        head_dir = RESULTS / "heads"
        head_dir.mkdir(parents=True, exist_ok=True)
        torch.save(head.cpu().state_dict(), head_dir / f"{name}.pt")
        (RESULTS / f"{name}_head_history.json").write_text(json.dumps(history, indent=2))
        del model, head
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _load_head(name, dim, classes, device):
    head = nn.Linear(dim, len(classes)).to(device)
    head.load_state_dict(torch.load(RESULTS / "heads" / f"{name}.pt",
                                    map_location=device, weights_only=True))
    return head.eval()


def _select_projection(ids, labels, per_class):
    # IDs are sorted and fixed in the saved manifest; no post-result sampling.
    selected = []
    for label in sorted(set(labels)):
        selected.extend(np.flatnonzero(np.asarray(labels) == label)[:per_class].tolist())
    return np.asarray(sorted(selected))


def evaluate_command(config, manifest, device):
    dataset = load_dataset(DATA, "test")
    ids = manifest["test_ids"]
    labels = np.asarray(dataset.labels)[ids]
    images = [image224(dataset, i) for i in ids]
    all_variants = {}
    for i, (image_id, image) in enumerate(zip(ids, images)):
        for key, transformed in variants(image, image_id, config["seed"], config["hue_degrees"]):
            all_variants.setdefault(key, []).append(transformed)
    cue_rows = accepted(RESULTS)
    cue_images = []
    for row in cue_rows:
        with Image.open(RESULTS / "cue_candidates" / row["candidate_file"]) as im:
            cue_images.append(im.convert("RGB").copy())
    class_to_id = {name: i for i, name in enumerate(manifest["classes"])}
    content_labels = np.asarray([class_to_id[r["content_class"]] for r in cue_rows])
    style_labels = np.asarray([class_to_id[r["style_class"]] for r in cue_rows])
    content_positions = np.asarray([ids.index(int(r["content_id"])) for r in cue_rows])
    projection_positions = _select_projection(ids, labels, config["representation_per_class"])
    output = {"config": config, "test_ids": ids, "models": {},
              "cue_review_counts": json.loads((RESULTS / "cue_review_counts.json").read_text())}
    translation_plot = {}
    for name in NAMES:
        model = FrozenBackbone(name, device)
        head = _load_head(name, model.dim, manifest["classes"], device)
        texts, scale = model.zero_shot(manifest["classes"]) if name == "clip_vit_b_32" else (None, None)
        features = {key: batched_features(model, ims, config["batch_size"], f"{name} {key}")
                    for key, ims in all_variants.items()}
        features["cue"] = batched_features(model, cue_images, config["batch_size"], name + " cue")
        heads = {name: head}
        if texts is not None:
            heads["clip_zero_shot"] = None
        for variant_name, classifier in heads.items():
            probabilities = {
                key: (probability(classifier, x, device) if classifier is not None else
                      zero_shot_probability(x, texts, scale, device))
                for key, x in features.items()
            }
            clean_p = probabilities["clean"]
            clean_metrics = metrics(labels, clean_p)
            model_result = {"clean": clean_metrics, "interventions": {}, "translation": {},
                            "cue_conflicts": {}, "representation_stability": {}}
            for key in ("grayscale", "hue", "patch"):
                m = metrics(labels, probabilities[key])
                m["accuracy_delta"] = m["accuracy"] - clean_metrics["accuracy"]
                m["consistency"] = consistency(clean_p, probabilities[key])
                model_result["interventions"][key] = m
            model_result["translation"]["0"] = {
                "accuracy": clean_metrics["accuracy"], "consistency": 1.0}
            for delta in (8, 16, 32):
                keys = [f"translate_{delta}_{d}" for d in ("right", "left", "down", "up")]
                model_result["translation"][str(delta)] = {
                    "accuracy": float(np.mean([metrics(labels, probabilities[k])["accuracy"] for k in keys])),
                    "consistency": float(np.mean([consistency(clean_p, probabilities[k]) for k in keys])),
                }
            cue_p = probabilities["cue"]
            model_result["cue_conflicts"] = cue_counts(
                cue_p.argmax(axis=1), content_labels, style_labels)
            model_result["cue_conflicts"]["by_pair_direction"] = {}
            for pair_direction in sorted({(r["pair"], r["direction"]) for r in cue_rows}):
                selected = np.asarray([(r["pair"], r["direction"]) == pair_direction for r in cue_rows])
                model_result["cue_conflicts"]["by_pair_direction"]["_".join(map(str, pair_direction))] = (
                    cue_counts(cue_p.argmax(1)[selected], content_labels[selected], style_labels[selected]))
            for key in ("grayscale", "patch"):
                model_result["representation_stability"][key] = cosine_stability(features["clean"], features[key])
            model_result["representation_stability"]["cue"] = cosine_stability(
                features["clean"][content_positions], features["cue"])
            for delta in (8, 16, 32):
                model_result["representation_stability"][f"translation_{delta}"] = float(np.mean([
                    cosine_stability(features["clean"], features[f"translate_{delta}_{direction}"])
                    for direction in ("right", "left", "down", "up")]))
            # Store per-image decisions for later failure analysis.
            predictions = []
            for row, pred, confidence in zip(cue_rows, cue_p.argmax(1), cue_p.max(1)):
                predictions.append({"id": row["id"], "content": row["content_class"],
                                    "style": row["style_class"], "prediction": manifest["classes"][pred],
                                    "confidence": float(confidence)})
            (RESULTS / f"{variant_name}_cue_predictions.json").write_text(json.dumps(predictions, indent=2))
            output["models"][variant_name] = model_result
            translation_plot[variant_name] = model_result["translation"]
        # Each plot fits clean and one transformed condition together. Zero-shot shares CLIP features.
        for key in ("grayscale", "patch", "translate_32_right"):
            plot_projection(features["clean"][projection_positions], features[key][projection_positions],
                            labels[projection_positions], manifest["classes"],
                            RESULTS / f"{name}_{key}_tsne.png", config["seed"], config["tsne_perplexity"])
        cue_plot_ids = np.arange(min(len(cue_rows), 200))
        plot_projection(features["clean"][content_positions[cue_plot_ids]],
                        features["cue"][cue_plot_ids], content_labels[cue_plot_ids],
                        manifest["classes"], RESULTS / f"{name}_cue_tsne.png",
                        config["seed"], config["tsne_perplexity"])
        del model, head, features
        if device.type == "cuda":
            torch.cuda.empty_cache()
    plot_translation(translation_plot, RESULTS / "translation.png")
    # One row per reviewed example makes cross-model agreements and failures easy to inspect.
    prediction_maps = {
        name: {row["id"]: row for row in json.loads(
            (RESULTS / f"{name}_cue_predictions.json").read_text())}
        for name in (*NAMES, "clip_zero_shot")
    }
    with (RESULTS / "cue_comparison.csv").open("w", newline="") as file:
        fieldnames = ["id", "content", "style", *NAMES, "clip_zero_shot"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in cue_rows:
            writer.writerow({"id": row["id"], "content": row["content_class"],
                             "style": row["style_class"],
                             **{name: prediction_maps[name][row["id"]]["prediction"]
                                for name in prediction_maps}})
    (RESULTS / "metrics.json").write_text(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "conflicts", "train", "evaluate"))
    parser.add_argument("--adain-repo", type=Path)
    parser.add_argument("--vgg-weights", type=Path)
    parser.add_argument("--decoder-weights", type=Path)
    parser.add_argument("--model", choices=NAMES, help="Train just one backbone's linear head")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    seed_all(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(DATA, RESULTS, config["seed"], config["test_per_class"])
        return
    manifest = json.loads((RESULTS / "splits.json").read_text())
    if args.command == "conflicts":
        if not all((args.adain_repo, args.vgg_weights, args.decoder_weights)):
            parser.error("conflicts requires --adain-repo, --vgg-weights, --decoder-weights")
        count = generate(DATA, RESULTS, args.adain_repo, args.vgg_weights,
                         args.decoder_weights, config, device)
        print(f"Generated {count} candidates; review cue_review.csv and contact sheets before evaluation")
    elif args.command == "train":
        train_command(config, manifest, device, args.model)
    elif args.command == "evaluate":
        evaluate_command(config, manifest, device)


if __name__ == "__main__":
    main()
