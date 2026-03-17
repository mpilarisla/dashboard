from pathlib import Path
import pandas as pd

# =========================================================
# RUTAS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
PROCESADOS_DIR = BASE_DIR / "procesados"
PROCESADOS_DIR.mkdir(exist_ok=True)

# =========================================================
# LEER TABLAS
# =========================================================

nbi_total = pd.read_csv(DATOS_DIR / "nbi_total_radio_gpm.csv")
nbi_ind = pd.read_csv(DATOS_DIR / "nbi_indicadores_radio_gpm.csv")

# asegurar tipo numérico
nbi_total["radio_id"] = pd.to_numeric(nbi_total["radio_id"], errors="coerce")
nbi_ind["radio_id"] = pd.to_numeric(nbi_ind["radio_id"], errors="coerce")

nbi_total = nbi_total.dropna(subset=["radio_id"]).copy()
nbi_ind = nbi_ind.dropna(subset=["radio_id"]).copy()

nbi_total["radio_id"] = nbi_total["radio_id"].astype(int)
nbi_ind["radio_id"] = nbi_ind["radio_id"].astype(int)

# =========================================================
# UNIR TABLAS
# =========================================================

nbi = nbi_total.merge(
    nbi_ind,
    on="radio_id",
    how="left"
)

# =========================================================
# CALCULAR TOTALES
# =========================================================

nbi["hogares_totales"] = (
    nbi["hogares_con_nbi"] +
    nbi["hogares_sin_nbi"]
)

nbi["prop_con_nbi"] = (
    nbi["hogares_con_nbi"] /
    nbi["hogares_totales"]
)

# =========================================================
# GUARDAR
# =========================================================

salida = PROCESADOS_DIR / "nbi_radio_gpm.csv"

nbi.to_csv(
    salida,
    index=False,
    encoding="utf-8"
)

print("Tabla NBI generada:")
print(salida)
print(nbi.head())