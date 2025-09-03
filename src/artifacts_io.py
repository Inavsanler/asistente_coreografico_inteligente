import os
import pathlib

# Conjuntos válidos de artefactos (elige el primero que exista)
REQUIRED_SETS = [
    ["complete_model_thresholded_bundle.joblib"],                           # preferente (bundle)
    ["pipeline_ovr_logreg.joblib", "feature_cols.csv", "label_names.csv"], # estándar
]
OPTIONAL_FILES = ["thresholds.json"]  # si lo usas

# 'requests' es opcional; si no está, se usa urllib (stdlib)
try:
    import requests as _requests
except Exception:  # ModuleNotFoundError u otros
    _requests = None
    import contextlib
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError

def _have_files(dirpath, files):
    return all(os.path.exists(os.path.join(dirpath, f)) for f in files)

def _download(url, dst, timeout=30):
    """
    Descarga con 'requests' si está instalado; si no, usa urllib de la stdlib.
    Lanza excepción si falla.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if _requests is not None:
        r = _requests.get(url, timeout=timeout)
        r.raise_for_status()
        with open(dst, "wb") as f:
            f.write(r.content)
    else:
        with contextlib.closing(urlopen(url, timeout=timeout)) as r:
            # En urllib, status puede no estar en algunas versiones; si existe y es >=400, error
            status = getattr(r, "status", 200)
            if status and int(status) >= 400:
                raise HTTPError(url, status, "HTTP error", hdrs=None, fp=None)
            data = r.read()
        with open(dst, "wb") as f:
            f.write(data)

def ensure_from_release(owner, repo, tag, files, target_dir="artifacts"):
    """Descarga 'files' desde el Release si no existen localmente."""
    base = f"https://github.com/{owner}/{repo}/releases/download/{tag}"
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    for fname in files:
        dst = os.path.join(target_dir, fname)
        if not os.path.exists(dst):
            _download(f"{base}/{fname}", dst)

def detect_variant(target_dir="artifacts"):
    """Devuelve la lista de archivos que forman el primer conjunto encontrado."""
    for fset in REQUIRED_SETS:
        if _have_files(target_dir, fset):
            return fset
    return None

def init_artifacts(owner=None, repo=None, tag=None, target_dir="artifacts"):
    """
    1) Si hay owner/repo/tag → intenta descargar de Releases (best effort).
    2) Detecta qué conjunto existe localmente (bundle o estándar).
    Devuelve dict con info de estado.
    """
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    release_attempted = False
    release_error = None

    if owner and repo and tag:
        release_attempted = True
        try:
            # Probar ambos conjuntos + opcionales; ignorar fallos parciales
            for fset in REQUIRED_SETS:
                try:
                    ensure_from_release(owner, repo, tag, fset, target_dir)
                except Exception:
                    pass
            try:
                ensure_from_release(owner, repo, tag, OPTIONAL_FILES, target_dir)
            except Exception:
                pass
        except Exception as e:
            release_error = str(e)

    variant = detect_variant(target_dir)
    return {
        "target_dir": target_dir,
        "variant": variant,                # None si no hay artefactos válidos
        "release_attempted": release_attempted,
        "release_error": release_error,    # None si no hubo error global
    }