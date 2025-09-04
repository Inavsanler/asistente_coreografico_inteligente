# src/model_io.py
import os, joblib, pandas as pd, numpy as np

def load_bundle(art_dir: str):
    """
    Espera un joblib con un dict:
    {
        "pipeline": sklearn-like estimator con .predict() (umbral incluido),
        "feature_cols": list[str],
        "label_names": list[str]
    }
    """
    path = os.path.join(art_dir, "complete_model_thresholded_bundle.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe: {path}")
    bundle = joblib.load(path)
    # Validaciones mínimas
    for k in ("pipeline", "feature_cols", "label_names"):
        if k not in bundle:
            raise ValueError(f"Falta clave '{k}' en el bundle.")
    return bundle

def ensure_feature_frame(feats: dict, feature_cols: list[str]) -> pd.DataFrame:
    """
    Alinea el dict de métricas (feats) a las columnas esperadas por el modelo.
    Rellena con NaN lo que falte y castea a float.
    """
    row = {c: np.nan for c in feature_cols}
    for k, v in feats.items():
        if k in row:
            try:
                row[k] = float(v)
            except Exception:
                row[k] = np.nan
    return pd.DataFrame([row], columns=feature_cols)
