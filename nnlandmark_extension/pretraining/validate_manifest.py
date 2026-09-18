import json
import SimpleITK as sitk

manifest_path = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/pretraining/manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

good, bad = [], []
for entry in manifest:
    try:
        img = sitk.ReadImage(entry["path"])
        _ = sitk.GetArrayFromImage(img)  # force full read, not just header
        good.append(entry)
    except Exception as e:
        print(f"BAD: {entry['path']} -- {e}")
        bad.append(entry)

print(f"\nGood: {len(good)}, Bad: {len(bad)}")

with open(manifest_path, "w") as f:
    json.dump(good, f, indent=2)
print(f"Manifest updated: {manifest_path} now has {len(good)} entries")
