
import numpy as np
def sugerencias_reglas(feats: dict):
    S=[]
    if feats.get("amplitud_x",0)<0.08 and feats.get("amplitud_y",0)<0.08:
        S.append("Aumenta la amplitud (extensiones y desplazamientos más amplios).")
    if feats.get("velocidad_media",0)<0.5:
        S.append("Incrementa la energía con acentos/diagonales.")
    sim = feats.get("simetria", np.nan)
    if np.isfinite(sim) and sim < 0.05:
        S.append("Rompe la simetría con contrastes o canons.")
    if not S:
        S.append("Buen equilibrio; prueba micro-variaciones de ritmo/foco.")
    return S
