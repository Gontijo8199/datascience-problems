import pandas as pd

local = "/home/rafa/programacao/datavis/a1-pinho/MICRODADOS_ENEM_2023.csv"
interno = "/home/rafa/programacao/datavis/a1-pinho/MICRODADOS_ENEM_2023_minas.csv"

dataminas = pd.read_csv(interno)

print(dataminas)
'''
def filtro(item):
    return item[item['CO_UF_ESC'] == 31]

first_chunk = True
for chunk in pd.read_csv(local, sep=';', encoding='ISO 8859-1', chunksize=100000):
    filtrado = filtro(chunk)
    if not filtrado.empty:
        filtrado.to_csv(interno, mode='a', header=first_chunk, index=False)
        first_chunk = False

''' 