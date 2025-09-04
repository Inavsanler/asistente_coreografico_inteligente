# src/suggestions.py
"""
Mapeo de etiquetas -> textos de sugerencia, con resolución de conflictos
entre etiquetas mutuamente excluyentes.
"""

from typing import List

# Textos base (ajusta a tus etiquetas reales del bundle)
TEXTS = {
    "amplitud_baja": "Proyecta más en horizontal y vertical para ganar presencia escénica.",
    "amplitud_excesiva": "Contén la amplitud para mejorar la precisión y el control.",
    "variedad_baja": "Introduce cambios de dirección, diagonales y contratiempos.",
    "variedad_excesiva": "Reduce la fragmentación; prioriza transiciones coherentes.",
    "poca_simetria": "Equilibra ambos lados del cuerpo para reforzar la limpieza.",
    "mucha_simetria": "Explora asimetrías para enriquecer la composición.",
    "fluidez_baja": "Trabaja transiciones encadenadas para ganar continuidad.",
    "poco_rango_niveles": "Incorpora niveles alto y bajo además del medio.",
    "exceso_rango_niveles": "Modula los niveles para sostener la narrativa.",
    # Añade aquí el resto de etiquetas de tu training
}

# Grupos mutuamente excluyentes (prioridad por orden)
MUTEX = [
    ["amplitud_baja", "amplitud_excesiva"],
    ["variedad_baja", "variedad_excesiva"],
    ["poca_simetria", "mucha_simetria"],
    ["poco_rango_niveles", "exceso_rango_niveles"],
]

def _resolve_mutex(labels: List[str]) -> List[str]:
    keep = set(labels)
    for g in MUTEX:
        inter = [x for x in g if x in keep]
        if len(inter) > 1:
            # Mantén la primera por prioridad
            for x in inter[1:]:
                keep.discard(x)
    return [x for x in labels if x in keep]

def map_labels_to_suggestions(labels_on: List[str]) -> List[str]:
    if not labels_on:
        return []
    labels_clean = _resolve_mutex(labels_on)
    out = []
    for lab in labels_clean:
        if lab in TEXTS:
            out.append(TEXTS[lab])
    return out
