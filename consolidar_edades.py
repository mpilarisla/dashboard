import pandas as pd
import geopandas as gpd
import glob
import os

# ----------------------------
# 1. Leer radios censales
# ----------------------------
radios = gpd.read_file("datos/radios-censales.shp")
gpm = radios[radios["NOMDEPTO"] == "GENERAL PUEYRREDON"].copy()
gpm["radio_id"] = gpm["LINK"].astype(str).str.zfill(9)

print(f"Radios en General Pueyrredón: {len(gpm)}")

# ----------------------------
# 2. Leer todos los CSV
# ----------------------------
archivos = glob.glob(os.path.join("datos", "resultados", "*.csv"))
print(f"CSV encontrados: {len(archivos)}")

lista = []

for archivo in archivos:
    df = pd.read_csv(archivo, encoding="latin1")

    # renombrar columnas
    df = df.rename(columns={
        "Radio": "radio_original",
        "Edad.en.grupos.quinquenales": "edad",
        "X__tot__": "total",
        "Mujer...Femenino": "mujeres",
        "VarÃ³n...Masculino": "varones"
    })

    # crear id compatible con shapefile
    df["radio_id"] = df["radio_original"].astype(str).str.zfill(9)

    # quedarnos con lo importante
    df = df[["radio_id", "edad", "total", "mujeres", "varones"]]

    lista.append(df)

# unir todos los CSV
edades = pd.concat(lista, ignore_index=True)

# ----------------------------
# 3. Limpiar datos básicos
# ----------------------------
edades["total"] = pd.to_numeric(edades["total"], errors="coerce")
edades["mujeres"] = pd.to_numeric(edades["mujeres"], errors="coerce")
edades["varones"] = pd.to_numeric(edades["varones"], errors="coerce")

# ----------------------------
# 4. Quedarnos solo con radios de GPM
# ----------------------------
edades_gpm = edades[edades["radio_id"].isin(gpm["radio_id"])].copy()

print(f"Filas de edades en GPM: {len(edades_gpm)}")
print("\nPrimeras filas:")
print(edades_gpm.head(10))

print("\nCantidad de radios únicos en edades_gpm:")
print(edades_gpm['radio_id'].nunique())

# ----------------------------
# 5. Guardar consolidado
# ----------------------------
os.makedirs("procesados", exist_ok=True)
edades_gpm.to_csv("procesados/edades_gpm.csv", index=False, encoding="utf-8")

print("\nArchivo guardado en: procesados/edades_gpm.csv")