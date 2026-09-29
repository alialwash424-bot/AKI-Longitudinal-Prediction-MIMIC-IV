import os
import pandas as pd
import file_system

folder = file_system.pick_directory()

source = os.path.join(folder, "labevents.csv.gz")
output = os.path.join(folder, "creatinine_events.csv")

itemids = {50912, 52546}

columns = [
    "subject_id",
    "hadm_id",
    "specimen_id",
    "itemid",
    "charttime",
    "storetime",
    "valuenum",
    "valueuom",
    "ref_range_lower",
    "ref_range_upper",
    "flag",
    "priority"
]

if not os.path.exists(source):
    raise FileNotFoundError("labevents.csv.gz was not found")

first_write = True
total_rows = 0
chunk_number = 0

for chunk in pd.read_csv(
    source,
    usecols=columns,
    chunksize=250000,
    low_memory=False
):
    chunk_number += 1
    selected = chunk[
        chunk["itemid"].isin(itemids)
        & chunk["valuenum"].notna()
    ].copy()

    if not selected.empty:
        selected.to_csv(
            output,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )
        first_write = False
        total_rows += len(selected)

    if chunk_number % 10 == 0:
        print("Chunks processed:", chunk_number)
        print("Creatinine rows found:", total_rows)

print("EXTRACTION COMPLETE")
print("Total creatinine rows:", total_rows)
print("Output:", output)
