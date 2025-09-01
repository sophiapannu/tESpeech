from textgrid import TextGrid
import pandas as pd
from pathlib import Path

#config 
TEXTGRID_PATH = Path("harvard.TextGrid")
TIER_NAME = "phones"  #the tier 
#load
tg = TextGrid.fromFile(str(TEXTGRID_PATH))
tier = next((t for t in tg.tiers if t.name == TIER_NAME), None)
if tier is None:
    raise SystemExit(f"Tier '{TIER_NAME}' not found. Available: {[t.name for t in tg.tiers]}")
rows = []
for it in tier.intervals:
    start = float(it.minTime)
    end = float(it.maxTime)
    label = (it.mark or "").strip()
    dur = max(0.0, end - start)
    rows.append({"start_s": start, "end_s": end, "dur_s": dur, "label": label})
df = pd.DataFrame(rows)
#show first 10 rows in terminal
print(df.head(10).to_string(index=False))
#save to CSV
out = TEXTGRID_PATH.with_suffix("").name + "_step1_table.csv"
df.to_csv(out, index=False)
print(f"\nSaved {out}")
