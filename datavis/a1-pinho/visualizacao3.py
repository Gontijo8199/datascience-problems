import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.figure(figsize=(14, 12))

colunas = ['NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']
dados = pd.read_csv("datascience-problems/datavis/a1-pinho/MICRODADOS_ENEM_2023.csv", 
                   sep=";", encoding="latin1", usecols=colunas, nrows=100000)

dados_filtrados = dados[dados['NU_NOTA_MT'] > 850].dropna()

materias = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']
titulos = ['Ciências da Natureza', 'Ciências Humanas', 'Linguagens e Códigos', 'Redação']

for i, (materia, titulo) in enumerate(zip(materias, titulos), 1):
    plt.subplot(2, 2, i)
    
    sns.scatterplot(data=dados_filtrados, x='NU_NOTA_MT', y=materia, alpha=0.6, color='royalblue')
    
    sns.regplot(data=dados_filtrados, x='NU_NOTA_MT', y=materia, 
                scatter=False, color='red', line_kws={'linestyle':'--'})
    
    corr = np.corrcoef(dados_filtrados['NU_NOTA_MT'], dados_filtrados[materia])[0, 1]
    
    plt.title(f'Nota em Matemática vs {titulo}\n(Corr: {corr:.2f})', pad=15)
    plt.xlabel('Nota em Matemática (> 850)')
    plt.ylabel(f'Nota em {titulo}')
    plt.xlim(850, dados_filtrados['NU_NOTA_MT'].max() + 10)
    plt.ylim(dados_filtrados[materia].min() - 10, dados_filtrados[materia].max() + 10)
    
    plt.axhline(y=850, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=850, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()