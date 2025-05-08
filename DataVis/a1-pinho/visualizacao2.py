import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cols = ['Q006', 'Q005', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
dados = pd.read_csv("datascience-problems/datavis/a1-pinho/MICRODADOS_ENEM_2023.csv", 
                   sep=";", encoding="latin1", usecols=cols)

dados['NOTA_MEDIA'] = dados[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']].mean(axis=1)
dados['Q005'] = pd.to_numeric(dados['Q005'], errors='coerce').clip(lower=1)

faixas_renda = {
    'B': (0, 1320), 'C': (1320, 1980), 'D': (1980, 2640), 'E': (2640, 3300),
    'F': (3300, 3960), 'G': (3960, 5280), 'H': (5280, 6600), 'I': (6600, 7920),
    'J': (7920, 9240), 'K': (9240, 10560), 'L': (10560, 11880), 'M': (11880, 13200),
    'N': (13200, 15840), 'O': (15840, 19800), 'P': (19800, 26400), 'Q': (26400, np.inf)
}

def classificar_renda_pc(row):
    if pd.isna(row['Q006']) or row['Q006'] == 'A':
        return None
    renda_min, renda_max = faixas_renda.get(row['Q006'], (0, 0))
    n_pessoas = row['Q005']
    sm = 1320
    renda_pc = renda_max / n_pessoas
    if renda_pc < sm:
        return 'Até 1 SM'
    elif renda_pc < 2*sm:
        return 'Entre 1-2 SM'
    elif renda_pc < 3*sm:
        return 'Entre 2-3 SM'
    elif renda_pc < 5*sm:
        return 'Entre 3-5 SM'
    else:
        return 'Acima de 5 SM'

dados['Per Capita'] = dados.apply(classificar_renda_pc, axis=1)
dados = dados.dropna(subset=['Per Capita', 'NOTA_MEDIA'])

ordem_renda = [
    'Até 1 SM',
    'Entre 1-2 SM',
    'Entre 2-3 SM',
    'Entre 3-5 SM',
    'Acima de 5 SM'
]

plt.figure(figsize=(16, 10))
ax = sns.kdeplot(
    data=dados,
    x='NOTA_MEDIA',
    hue='Per Capita',
    multiple='fill',
    common_norm=False,
    fill=True,
    alpha=0.85,
    linewidth=0.5,
    hue_order=ordem_renda,
    palette='viridis'
)

plt.title('Distribuição das Notas Médias do ENEM 2023 por Salário Mínimo per Capita', 
          fontsize=14, pad=20)

ax.set_ylabel('Proporção', fontsize=14, rotation=0, loc='top', labelpad=40)
ax.set_xlabel('Nota Média', fontsize=14, labelpad=10)

plt.xlim(300, 850)
plt.xticks(np.arange(300, 851, 50), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.2)

plt.tight_layout()
plt.show()
