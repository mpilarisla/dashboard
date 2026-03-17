import os
import pandas as pd

base = r"C:\Users\usuario\Documents\Proyectos\dashboard_generalpueyrredon\datos"

print("Archivos en datos:")
print(os.listdir(base))

print("\n--- VIVIENDAS ---")
viv = pd.read_csv(os.path.join(base, "viviendas_ocupacion_radio_gpm.csv"))
print(viv.columns.tolist())
print(viv.head())

print("\n--- NBI TOTAL ---")
nbi_total = pd.read_csv(os.path.join(base, "nbi_total_radio_gpm.csv"))
print(nbi_total.columns.tolist())
print(nbi_total.head())

print("\n--- NBI INDICADORES ---")
nbi_ind = pd.read_csv(os.path.join(base, "nbi_indicadores_radio_gpm.csv"))
print(nbi_ind.columns.tolist())
print(nbi_ind.head())