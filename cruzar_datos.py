import geopandas as gpd
import pandas as pd

# -------------------------
# 1. Leer radios censales
# -------------------------
radios = gpd.read_file("datos/radios-censales.shp")

# Filtrar General Pueyrredón
radios_gpm = radios[radios["NOMDEPTO"] == "GENERAL PUEYRREDON"].copy()

# Pasar LINK a número entero para que coincida con radio_id del CSV
radios_gpm["link_num"] = radios_gpm["LINK"].astype(int)

print("Cantidad de radios en shapefile GPM:", len(radios_gpm))
print(radios_gpm[["LINK", "link_num"]].head())

# -------------------------
# 2. Leer CSV exportado desde R
# -------------------------
edades = pd.read_csv("datos/edad_sexo_radio_gpm.csv")

print("\nColumnas del CSV:")
print(edades.columns.tolist())

print("\nPrimeras filas del CSV:")
print(edades.head())

# Asegurar que radio_id sea numérico
edades["radio_id"] = pd.to_numeric(edades["radio_id"], errors="coerce")

# -------------------------
# 3. Ver cuántos radios únicos hay en el CSV
# -------------------------
print("\nCantidad de radios únicos en CSV:", edades["radio_id"].nunique())

# -------------------------
# 4. Cruzar shapefile con CSV
# -------------------------
datos_cruzados = radios_gpm.merge(
    edades,
    left_on="link_num",
    right_on="radio_id",
    how="left"
)

print("\nCantidad de filas luego del cruce:", len(datos_cruzados))

# -------------------------
# 5. Verificar radios con datos
# -------------------------
radios_con_datos = datos_cruzados["radio_id"].notna().sum()
print("Filas con datos del CSV:", radios_con_datos)

print("\nEjemplo de cruce:")
print(
    datos_cruzados[
        ["LINK", "link_num", "radio_id", "edad_label", "sexo_label", "personas"]
    ].head(20)
)

# -------------------------
# 6. Guardar resultado si querés
# -------------------------
datos_cruzados.to_file("procesados/radios_edades_cruzado.geojson", driver="GeoJSON")
print("\nArchivo guardado en procesados/radios_edades_cruzado.geojson")