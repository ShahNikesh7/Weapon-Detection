from pathlib import Path
import csv

CSV_PATH = Path("captions.csv")  # UPDATE TO YOUR CSV PATH
OUT_YAML = Path("captions.yaml") # UPDATE TO WHERE YOUR IMAGES ARE

with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter=",")
    lines = ["items:"]
    for row in reader:
        fp = row["filepath"]
        cap = row["caption"]
        lines.append(f"  - filepath: {fp}")
        lines.append(f"    caption: {cap}")

yaml_text = "\n".join(lines) + "\n"
OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
OUT_YAML.write_text(yaml_text, encoding="utf-8")
print(f"[OK] wrote -> {OUT_YAML}")