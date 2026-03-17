import geopandas as gpd
import pandas as pd

# 1. Leer capa de radios
radios = gpd.read_file("datos/radios-censales.shp")

# 2. Filtrar General Pueyrredón
gpm = radios[radios["NOMDEPTO"] == "GENERAL PUEYRREDON"].copy()

# 3. Crear clave como texto
gpm["link_str"] = gpm["LINK"].astype(str).str.zfill(9)

print("Cantidad de radios en General Pueyrredón:", len(gpm))
print("\nPrimeros LINK de GPM:")
print(gpm["link_str"].head(10).tolist())

# 4. Leer un CSV de ejemplo
df = pd.read_csv("datos/resultados/radio_0635773_63577301.csv", encoding="latin1")

# 5. Crear clave comparable desde el CSV
df["radio_str"] = df["Radio"].astype(str).str.zfill(9)

radio_csv = df["radio_str"].iloc[0]
print("\nRadio del CSV transformado:", radio_csv)

# 6. Buscar si existe en la capa de radios
coincidencia = gpm[gpm["link_str"] == radio_csv]

print("\nCantidad de coincidencias encontradas:", len(coincidencia))

if len(coincidencia) > 0:
    print("\nCoincidencia encontrada:")
    print(coincidencia[["NOMDEPTO", "FRAC", "RADIO", "LINK", "link_str"]].head())
else:
    print("\nNo se encontró coincidencia.")