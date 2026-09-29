import os
from collections import Counter

import pandas as pd
import file_system

folder = file_system.pick_directory()

source = os.path.join(folder, "outputevents.csv.gz")
output = os.path.join(folder, "urine_output_events.csv")
report = os.path.join(folder, "urine_output_audit.txt")

if not os.path.exists(source):
    raise FileNotFoundError(
        "outputevents.csv.gz was not found. "
        "Make sure its exact filename is outputevents.csv.gz"
    )

# MIMIC-IV urine-output item IDs
item_labels = {
    226557: "Right ureteral stent",
    226558: "Left ureteral stent",
    226559: "Foley catheter",
    226560: "Void",
    226561: "Condom catheter",
    226563: "Suprapubic catheter",
    226564: "Right nephrostomy",
    226565: "Left nephrostomy",
    226567: "Straight catheter",
    226584: "Ileal conduit",
    227488: "GU irrigant volume in",
    227489: "GU irrigant/urine volume out",
}

urine_itemids = set(item_labels)

columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "charttime",
    "itemid",
    "value",
    "valueuom",
]

first_write = True
chunk_number = 0
total_rows = 0
invalid_raw_values = 0
extreme_raw_values = 0
irrigant_input_rows = 0

unique_patients = set()
unique_stays = set()
item_counts = Counter()
unit_counts = Counter()

for chunk in pd.read_csv(
    source,
    usecols=columns,
    chunksize=250000,
    low_memory=False
):
    chunk_number += 1

    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")

    selected = chunk[
        chunk["itemid"].isin(urine_itemids)
        & chunk["stay_id"].notna()
        & chunk["charttime"].notna()
        & chunk["value"].notna()
    ].copy()

    if not selected.empty:
        selected["subject_id"] = selected["subject_id"].astype("int64")
        selected["stay_id"] = selected["stay_id"].astype("int64")
        selected["itemid"] = selected["itemid"].astype("int64")

        selected["raw_value"] = selected["value"]
        selected["urine_ml"] = selected["value"]

        # Irrigant input must be subtracted from urine output
        irrigant_mask = selected["itemid"] == 227488
        selected.loc[irrigant_mask, "urine_ml"] = (
            -selected.loc[irrigant_mask, "raw_value"]
        )

        selected["item_label"] = selected["itemid"].map(item_labels)

        selected = selected[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "charttime",
                "itemid",
                "item_label",
                "raw_value",
                "urine_ml",
                "valueuom",
            ]
        ]

        selected.to_csv(
            output,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

        total_rows += len(selected)
        invalid_raw_values += int((selected["raw_value"] <= 0).sum())
        extreme_raw_values += int((selected["raw_value"] >= 5000).sum())
        irrigant_input_rows += int(irrigant_mask.sum())

        unique_patients.update(selected["subject_id"].tolist())
        unique_stays.update(selected["stay_id"].tolist())
        item_counts.update(selected["itemid"].tolist())

        for itemid, unit in zip(
            selected["itemid"],
            selected["valueuom"].fillna("MISSING"),
        ):
            unit_counts[(int(itemid), str(unit))] += 1

    if chunk_number % 10 == 0:
        print("Chunks processed:", chunk_number)
        print("Urine-output rows found:", total_rows)

lines = [
    "URINE OUTPUT EXTRACTION COMPLETE",
    f"Total selected rows: {total_rows}",
    f"Unique patients: {len(unique_patients)}",
    f"Unique ICU stays: {len(unique_stays)}",
    f"GU irrigant input rows: {irrigant_input_rows}",
    f"Raw values <= 0: {invalid_raw_values}",
    f"Raw values >= 5000: {extreme_raw_values}",
    "",
    "ROWS BY ITEMID:",
]

for itemid in sorted(item_labels):
    lines.append(
        f"{itemid} | {item_labels[itemid]}: {item_counts[itemid]}"
    )

lines.append("")
lines.append("ROWS BY ITEMID AND UNIT:")

for key, count in sorted(unit_counts.items()):
    lines.append(f"{key[0]} | {key[1]}: {count}")

with open(report, "w") as file:
    file.write("\n".join(lines))

print()
print("\n".join(lines))
print()
print("Data output:", output)
print("Audit report:", report)
