import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

colunas = [
    'TP_DEPENDENCIA_ADM_ESC', 
    'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO'
]

df = pd.read_csv('datascience-problems/datavis/a1-pinho/MICRODADOS_ENEM_2023.csv', 
                 sep=';', usecols=colunas, encoding='latin1')
df = df.dropna()

df['Tipo de Escola'] = df['TP_DEPENDENCIA_ADM_ESC'].apply(lambda x: 'Pública' if x in [1, 2, 3] else 'Particular')

disciplinas = {
    'NU_NOTA_CN': 'Ciências da Natureza',
    'NU_NOTA_CH': 'Ciências Humanas',
    'NU_NOTA_LC': 'Linguagens',
    'NU_NOTA_MT': 'Matemática',
    'NU_NOTA_REDACAO': 'Redação'
}

df_long = df.melt(id_vars=['Tipo de Escola'], 
                  value_vars=disciplinas.keys(), 
                  var_name='Disciplina', 
                  value_name='Nota')

df_long['Disciplina'] = df_long['Disciplina'].map(disciplinas)

cor_publica = '#fae564' 
cor_particular = '#605795' 

plt.figure(figsize=(12, 6))
ax = sns.violinplot(
    data=df_long, 
    x='Disciplina', 
    y='Nota', 
    hue='Tipo de Escola', 
    split=True, 
    inner=None,  
    palette=[cor_publica, cor_particular]
)

plt.yticks(np.arange(0, 1100, 100), fontsize=18)
plt.xticks(fontsize=16)

ax.set_ylabel('Nota', fontsize=18, rotation=0, loc='top')
ax.set_xlabel('Disciplina', fontsize=18, labelpad=10)

disciplinas_unicas = df_long['Disciplina'].unique()
x_positions = np.arange(len(disciplinas_unicas))

for i, disciplina in enumerate(disciplinas_unicas):
    subset = df_long[df_long['Disciplina'] == disciplina]
    notas_publica = subset[subset['Tipo de Escola'] == 'Pública']['Nota']
    notas_particular = subset[subset['Tipo de Escola'] == 'Particular']['Nota']
    
    kde_publica = gaussian_kde(notas_publica)
    kde_particular = gaussian_kde(notas_particular)
    
    y = np.linspace(0, 1000, 300)
    densidade_publica = kde_publica(y)
    densidade_particular = kde_particular(y)
    
    max_density = max(densidade_publica.max(), densidade_particular.max())
    diff = (densidade_particular - densidade_publica) / (2 * max_density) * 0.4  
    
    plt.plot(i + diff, y, color='black', linewidth=1.5, label='Diferença de Densidade' if i == 0 else "")

plt.grid(True, linestyle='--', alpha=0.5)
plt.title('Distribuição das Notas do ENEM 2023 por Tipo de Escola e Prova', fontsize=20)
plt.ylim(0, 1000)


handles, labels = ax.get_legend_handles_labels()
line = Line2D([0], [0], color='black', lw=2, label='Diferença de Densidade')

handles.append(line)
plt.legend(handles=handles, labels=labels, title='Legenda', loc='upper left', fontsize=13)

plt.tight_layout()
plt.show()
