
import numpy as np
def features_coreograficos(K, fps=30.0):
    T,J,D = K.shape
    feats = {}
    with np.errstate(invalid="ignore"):
        amp = (np.nanmax(K, axis=0) - np.nanmin(K, axis=0)).mean(axis=0)
    feats["amplitud_x"] = float(amp[0]) if np.isfinite(amp[0]) else 0.0
    feats["amplitud_y"] = float(amp[1]) if np.isfinite(amp[1]) else 0.0
    if T>=2:
        disp = np.diff(K, axis=0)
        spd = np.linalg.norm(disp, axis=2)
        feats["velocidad_media"] = float(np.nanmean(spd) * fps)
    else:
        feats["velocidad_media"] = 0.0
    pairs = [(5,6),(11,12),(13,14),(15,16)]
    diffs=[]
    for L,R in pairs:
        if L < J and R < J:
            diffs.append(np.nanmean(np.linalg.norm(K[:,L,:]-K[:,R,:], axis=1)))
    feats["simetria"] = float(np.nanmean(diffs)) if diffs else np.nan
    feats["frames"] = int(T); feats["joints"] = int(J); feats["dims"] = int(D)
    return feats
