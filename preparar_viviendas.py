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
# LEER TABLA
# =========================================================

viv = pd.read_csv(DATOS_DIR / "viviendas_ocupacion_radio_gpm.csv")

# asegurar tipo numérico
viv["radio_id"] = pd.to_numeric(viv["radio_id"], errors="coerce")

viv = viv.dropna(subset=["radio_id"]).copy()
viv["radio_id"] = viv["radio_id"].astype(int)

# =========================================================
# PASAR A FORMATO ANCHO
# =========================================================

viv_wide = viv.pivot_table(
    index="radio_id",
    columns="ocup_cod",
    values="viviendas",
    aggfunc="sum",
    fill_value=0
).reset_index()

# renombrar columnas
viv_wide = viv_wide.rename(columns={
    1: "viviendas_ocupadas",
    2: "viviendas_desocupadas"
})

# asegurar columnas
if "viviendas_ocupadas" not in viv_wide.columns:
    viv_wide["viviendas_ocupadas"] = 0

if "viviendas_desocupadas" not in viv_wide.columns:
    viv_wide["viviendas_desocupadas"] = 0

# total
viv_wide["viviendas_totales"] = (
    viv_wide["viviendas_ocupadas"] +
    viv_wide["viviendas_desocupadas"]
)

# =========================================================
# GUARDAR
# =========================================================

salida = PROCESADOS_DIR / "viviendas_radio_gpm.csv"

viv_wide.to_csv(
    salida,
    index=False,
    encoding="utf-8"
)

print("Tabla de viviendas generada:")
print(salida)
print(viv_wide.head())