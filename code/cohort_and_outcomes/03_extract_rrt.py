import os
from collections import Counter

import pandas as pd
import file_system

folder = file_system.pick_directory()

source = os.path.join(folder, "procedureevents.csv.gz")
events_output = os.path.join(folder, "rrt_procedure_events.csv")
stays_output = os.path.join(folder, "rrt_stays.csv")
report_output = os.path.join(folder, "rrt_extraction_audit.txt")

if not os.path.exists(source):
    raise FileNotFoundError(
        "procedureevents.csv.gz was not found."
    )

rrt_labels = {
    225441: "Hemodialysis",
    225802: "CRRT",
    225803: "CVVHD",
    225805: "Peritoneal Dialysis",
    224270: "Dialysis Catheter",
    225809: "CVVHDF",
    225955: "SCUF",
    225436: "CRRT Filter Change",
}

# These represent active dialysis treatment
active_rrt_ids = {
    225441,
    225802,
    225803,
    225805,
    225809,
    225955,
}

# These are supporting evidence, not dialysis initiation alone
support_ids = {
    224270,
    225436,
}

columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "starttime",
    "endtime",
    "itemid",
    "value",
    "valueuom",
    "location",
    "locationcategory",
    "ordercategoryname",
    "patientweight",
    "statusdescription",
]

first_write = True
chunk_number = 0
total_rows = 0
active_rows = 0
support_rows = 0

unique_patients = set()
unique_stays = set()
active_stays = set()
item_counts = Counter()

for chunk in pd.read_csv(
    source,
    usecols=columns,
    chunksize=200000,
    low_memory=False
):
    chunk_number += 1

    chunk["itemid"] = pd.to_numeric(
        chunk["itemid"], errors="coerce"
    )

    selected = chunk[
        chunk["itemid"].isin(rrt_labels)
        & chunk["stay_id"].notna()
        & chunk["starttime"].notna()
    ].copy()

    if not selected.empty:
        selected["subject_id"] = selected[
            "subject_id"
        ].astype("int64")

        selected["stay_id"] = selected[
            "stay_id"
        ].astype("int64")

        selected["itemid"] = selected[
            "itemid"
        ].astype("int64")

        selected["rrt_label"] = selected[
            "itemid"
        ].map(rrt_labels)

        selected["active_rrt"] = selected[
            "itemid"
        ].isin(active_rrt_ids).astype(int)

        selected["supporting_evidence"] = selected[
            "itemid"
        ].isin(support_ids).astype(int)

        selected.to_csv(
            events_output,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        first_write = False

        total_rows += len(selected)
        active_rows += int(selected["active_rrt"].sum())
        support_rows += int(
            selected["supporting_evidence"].sum()
        )

        unique_patients.update(
            selected["subject_id"].tolist()
        )
        unique_stays.update(
            selected["stay_id"].tolist()
        )

        active_stays.update(
            selected.loc[
                selected["active_rrt"] == 1,
                "stay_id"
            ].tolist()
        )

        item_counts.update(
            selected["itemid"].tolist()
        )

    print("Chunks processed:", chunk_number)
    print("RRT-related rows:", total_rows)

if first_write:
    print("No RRT procedure records were found.")
    raise SystemExit

# Construct one summary row per ICU stay
rrt = pd.read_csv(
    events_output,
    parse_dates=["starttime", "endtime"],
    low_memory=False
)

active = rrt[rrt["active_rrt"] == 1].copy()

active_summary = active.groupby(
    ["subject_id", "hadm_id", "stay_id"],
    dropna=False
).agg(
    first_rrt_start=("starttime", "min"),
    last_rrt_end=("endtime", "max"),
    active_rrt_records=("active_rrt", "sum"),
    rrt_types=("rrt_label", lambda x: "|".join(sorted(set(x)))),
).reset_index()

support = rrt.groupby(
    "stay_id"
).agg(
    all_rrt_related_records=("itemid", "size"),
    supporting_records=("supporting_evidence", "sum"),
).reset_index()

stay_summary = active_summary.merge(
    support,
    on="stay_id",
    how="left"
)

stay_summary["received_active_rrt"] = 1
stay_summary.to_csv(stays_output, index=False)

lines = [
    "RRT EXTRACTION COMPLETE",
    f"Chunks processed: {chunk_number}",
    f"Total RRT-related rows: {total_rows}",
    f"Active RRT rows: {active_rows}",
    f"Supporting-evidence rows: {support_rows}",
    f"Unique patients with RRT evidence: {len(unique_patients)}",
    f"ICU stays with any RRT evidence: {len(unique_stays)}",
    f"ICU stays with active RRT: {len(active_stays)}",
    "",
    "ROWS BY ITEMID:",
]

for itemid in sorted(rrt_labels):
    lines.append(
        f"{itemid} | {rrt_labels[itemid]}: "
        f"{item_counts[itemid]}"
    )

with open(
    report_output,
    "w",
    encoding="utf-8",
    errors="replace"
) as file:
    file.write("\n".join(lines))

print()
print("RRT EXTRACTION COMPLETE")
print("Total RRT-related rows:", total_rows)
print("Active RRT rows:", active_rows)
print("Supporting-evidence rows:", support_rows)
print("Patients with RRT evidence:", len(unique_patients))
print("ICU stays with any RRT evidence:", len(unique_stays))
print("ICU stays with active RRT:", len(active_stays))
print("Event output:", events_output)
print("Stay output:", stays_output)
print("Audit report:", report_output)
