import csv
import json
from pathlib import Path

# Mapping of candidate IDs to rejection reasons.
# If an ID is in this dict, accept = "no" and reason = rejections[id].
# If an ID is NOT in this dict, accept = "yes" and reason = "".

rejections = {
    # Pair 0 Dir 0 (cat -> dog)
    "p0_d0_02": "severe artifacts and content outline not recognizable",
    "p0_d0_03": "content outline not recognizable",
    "p0_d0_05": "severe artifacts",
    "p0_d0_07": "content outline not recognizable",
    "p0_d0_09": "severe artifacts and outline distorted",
    "p0_d0_16": "severe artifacts",
    "p0_d0_24": "severe artifacts",
    "p0_d0_28": "severe artifacts",
    "p0_d0_37": "severe artifacts",

    # Pair 0 Dir 1 (dog -> cat)
    "p0_d1_06": "severe artifacts",
    "p0_d1_10": "severe artifacts",
    "p0_d1_12": "severe artifacts and content outline not recognizable",
    "p0_d1_13": "severe artifacts",
    "p0_d1_14": "severe artifacts",

    # Pair 1 Dir 0 (deer -> horse)
    "p1_d0_07": "severe artifacts",
    "p1_d0_10": "severe artifacts",
    "p1_d0_13": "severe artifacts",
    "p1_d0_17": "content outline not recognizable",
    "p1_d0_20": "severe artifacts",
    "p1_d0_21": "content outline not recognizable",

    # Pair 1 Dir 1 (horse -> deer)
    "p1_d1_04": "severe artifacts",
    "p1_d1_21": "severe artifacts",
    "p1_d1_23": "severe artifacts",
    "p1_d1_39": "severe artifacts",

    # Pair 2 Dir 0 (car -> truck)
    "p2_d0_03": "severe artifacts",
    "p2_d0_25": "content outline not recognizable",
    "p2_d0_30": "severe artifacts",

    # Pair 2 Dir 1 (truck -> car)
    "p2_d1_02": "severe artifacts",
    "p2_d1_06": "severe artifacts",
    "p2_d1_34": "content outline not recognizable",

    # Pair 3 Dir 0 (airplane -> bird)
    "p3_d0_01": "severe artifacts",
    "p3_d0_07": "severe artifacts and content outline not recognizable",
    "p3_d0_12": "content outline not recognizable",
    "p3_d0_19": "severe artifacts",
    "p3_d0_29": "severe artifacts",
    "p3_d0_30": "content outline not recognizable",

    # Pair 3 Dir 1 (bird -> airplane)
    "p3_d1_01": "content outline not recognizable",
    "p3_d1_04": "severe artifacts",
    "p3_d1_05": "severe artifacts",
    "p3_d1_14": "content outline not recognizable",
    "p3_d1_15": "content outline not recognizable",
    "p3_d1_21": "severe artifacts",
    "p3_d1_22": "severe artifacts",
    "p3_d1_33": "severe artifacts",
    "p3_d1_34": "content outline not recognizable",
    "p3_d1_39": "content outline not recognizable",

    # Pair 4 Dir 0 (airplane -> ship)
    "p4_d0_14": "severe artifacts",
    "p4_d0_19": "content outline not recognizable",
    "p4_d0_37": "content outline not recognizable",

    # Pair 4 Dir 1 (ship -> airplane)
    "p4_d1_32": "content outline not recognizable",
    "p4_d1_34": "content outline not recognizable",
    "p4_d1_36": "severe artifacts and content outline not recognizable",
}

csv_path = Path("cue_review_bundle/cue_review.csv")
with csv_path.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

print(f"Loaded {len(rows)} rows from {csv_path}")

counts = {}
for r in rows:
    uid = r["id"]
    if uid in rejections:
        r["accept"] = "no"
        r["reason"] = rejections[uid]
    else:
        r["accept"] = "yes"
        r["reason"] = ""

    grp = f"{r['pair']}_{r['direction']}"
    counts.setdefault(grp, {"accepted": 0, "rejected": 0, "total": 0})
    counts[grp]["total"] += 1
    if r["accept"] == "yes":
        counts[grp]["accepted"] += 1
    else:
        counts[grp]["rejected"] += 1

print("\nReview counts per group:")
for grp in sorted(counts):
    c = counts[grp]
    print(f"  Group {grp}: accepted={c['accepted']}, rejected={c['rejected']}, total={c['total']}")

# Write back to cue_review_bundle/cue_review.csv
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSuccessfully wrote updated annotations to {csv_path}")

# Also copy to task1/results/cue_review.csv if task1/results is needed
results_dir = Path("task1/results")
results_dir.mkdir(parents=True, exist_ok=True)
results_csv = results_dir / "cue_review.csv"
with results_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Also wrote copy to {results_csv}")

