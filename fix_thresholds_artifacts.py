# tools/fix_thresholds_artifacts.py
# Ejecuta:  python tools/fix_thresholds_artifacts.py  [--dir artifacts]
import os, json, argparse, shutil
from pathlib import Path
import joblib
import numpy as np

def safe_float(v, default=0.5):
    try:
        if isinstance(v, (int, float, np.floating, np.integer)):
            return float(v)
        if isinstance(v, str):
            return float(v.strip())
        if isinstance(v, dict):
            for k in ("thr", "value", "threshold", "umbral", "val"):
                if k in v:
                    return safe_float(v[k], default)
        return float(default)
    except Exception:
        return float(default)

def sanitize_map(maybe_map, default=0.5):
    if not isinstance(maybe_map, dict):
        return {}
    out = {}
    for k, v in maybe_map.items():
        out[str(k)] = safe_float(v, default)
    return out

def fix_json(path: Path):
    if not path.exists():
        return False, "not_found"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        fixed = sanitize_map(data, 0.5)
    elif isinstance(data, (list, tuple)):
        # si alguien guardó lista por índice
        fixed = {str(i): safe_float(v, 0.5) for i, v in enumerate(data)}
    else:
        return False, "invalid_json_type"

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)
    return True, "fixed"

def fix_bundle(path: Path):
    if not path.exists():
        return False, "not_found"
    obj = joblib.load(path)

    changed = False
    # nivel raíz
    if isinstance(obj, dict) and "thresholds" in obj:
        obj["thresholds"] = sanitize_map(obj["thresholds"], 0.5)
        changed = True

    # dentro de pipeline
    pipe = obj.get("pipeline") if isinstance(obj, dict) else None
    if pipe is not None:
        # algunos guardan thresholds en metadatos
        thr = getattr(pipe, "thresholds_", None)
        if thr is not None:
            setattr(pipe, "thresholds_", sanitize_map(thr, 0.5))
            changed = True

    if not changed:
        return False, "no_thresholds_field"

    out = path.with_name(path.stem + "_fixed" + path.suffix)
    joblib.dump(obj, out)
    return True, f"saved:{out.name}"

def main(art_dir: str):
    art = Path(art_dir)
    art.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Normalizando umbrales en: {art.resolve()}")

    for fname in ["thresholds.json", "complete_thresholds.json"]:
        ok, msg = fix_json(art / fname)
        if ok:
            print(f"  ✔ JSON {fname}: {msg}")
        else:
            print(f"  (skip) JSON {fname}: {msg}")

    ok, msg = fix_bundle(art / "complete_model_thresholded_bundle.joblib")
    if ok:
        print(f"  ✔ bundle: {msg}")
    else:
        print(f"  (skip) bundle: {msg}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts", help="Carpeta de artefactos")
    args = ap.parse_args()
    main(args.dir)
