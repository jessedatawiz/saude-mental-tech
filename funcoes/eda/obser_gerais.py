import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(lista_df: list) -> None:


    num_cols = []
    num_linhas = []
    anos = list(range(2016, 2022))

    num_cols = [df.shape[1] for df in lista_df]
    num_linhas = [df.shape[0] for df in lista_df]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Configurações estéticas
    cor_linhas = '#007ACC'
    cor_colunas = '#FF7F0E'
    largura_barras = 0.5

    # Gráfico de participantes por ano
    axs[0].bar(anos, num_linhas, color=cor_linhas, width=largura_barras)
    axs[0].set_ylabel('Número de Participantes')
    axs[0].set_title('Número de Participantes por Ano')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].tick_params(axis='x', bottom=False)

    # Gráfico de colunas por ano
    axs[1].bar(anos, num_cols, color=cor_colunas, width=largura_barras)
    axs[1].bar(anos, num_cols)
    axs[1].set_ylabel('Número de Colunas')
    axs[1].set_title('Número de Colunas por Ano')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].tick_params(axis='x', bottom=False)

    path_save = './arquivos/output/eda/observacoes_gerais.png'

    fig.savefig(path_save, dpi=300)

    plt.show()

    return None




                