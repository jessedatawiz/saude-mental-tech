import pandas as pd

def load_csv_files(file_paths):
    dataframes = []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    return dataframes

from googletrans import Translator
def translate(df, destination_language='pt'):
    # Inicializa o tradutor
    translator = Translator(service_urls=['translate.google.com'])

    # Traduz os nomes das colunas
    translated_columns = [translator.translate(column, dest=destination_language).text for column in df.columns]
    df.columns = translated_columns

    return df

# Compara o nome das colunas a cada par de dataframes.
def comparar_colunas(df1, df2):
    colunas_df1 = set(df1.columns)
    colunas_df2 = set(df2.columns)
    
    colunas_diferentes = colunas_df1.symmetric_difference(colunas_df2)
    
    if colunas_diferentes:
        print("Colunas que não são iguais: \n")
        for coluna in colunas_diferentes:
            if coluna in colunas_df1:
                print(f"- {coluna}: Encontrada no df1, mas não no df2 \n")
            else:
                print(f"- {coluna}: Encontrada no df2, mas não no df1 \n")
    else:
        print("Todos os nomes de colunas são iguais.")

# Compara em pares as colunas dos dataframes
def checar_cols(dfs):
    years = range(2017, 2022)
    
    for i, df1 in enumerate(dfs[:-1]):
        year1 = years[i]
        
        for j, df2 in enumerate(dfs[i+1:], i+1):
            year2 = years[j]
            
            print(f"{year1}/{year2}: \n")
            comparar_colunas(df1, df2)
            print()

# Verifica se a ordem das colunas é a mesma nos dataframes
def verificar_ordem_colunas(dataframes):
    # Obtém a lista de colunas do primeiro dataframe
    cols_ref = list(dataframes[0].columns)

    # Verifica a ordem das colunas em cada dataframe subsequente
    for df in dataframes[1:]:
        cols_atual = list(df.columns)

        # Verifica se as listas de colunas têm o mesmo comprimento
        if len(cols_atual) != len(cols_ref):
            return False

        # Verifica se a ordem das colunas é a mesma
        for col_ref, col_atual in zip(cols_ref, cols_atual):
            if col_ref != col_atual:
                return False

    # Se chegou até aqui, a ordem das colunas é a mesma em todos os dataframes
    return True

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

"""
Cria uma imagem com o número de participantes por ano e com o número de colunas por ano.
"""
def plot_num_cols(lista_df: list, path: str) -> None:

    num_cols = []
    num_cols = [df.shape[1] for df in lista_df]
    anos = list(range(2017, 2022))
    
    # Configurações estéticas
    cor = '#007ACC'
    largura_barras = 0.5

    # Gráfico de colunas por ano
    fig, ax = plt.subplots()
    ax.bar(anos, num_cols, color=cor, width=largura_barras)
    ax.set_ylabel('Número de Colunas')
    ax.set_title('Número de Colunas por Ano')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', bottom=False)

    # Adicionar números nas barras do gráfico de colunas
    for i, valor in enumerate(num_cols):
        ax.text(anos[i], valor, str(valor), ha='center', va='bottom', fontsize=8)
   
    plt.savefig(path, dpi=300)

    return None

def plot_num_participantes(lista_df: list, path: str) -> None:
    
    num_linhas = [df.shape[0] for df in lista_df]
    anos = list(range(2017, 2022))
    
    # Configurações estéticas
    cor = '#007ACC'
    largura_barras = 0.5

      # Gráfico de participantes por ano
    fig, ax = plt.subplots()
    ax.bar(anos, num_linhas, color=cor, width=largura_barras)
    ax.set_ylabel('Número de Participantes')
    ax.set_title('Número de Participantes por Ano')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', bottom=False)

    # Adicionar números nas barras do gráfico de participantes
    for i, valor in enumerate(num_linhas):
        ax.text(anos[i], valor, str(valor), ha='center', va='bottom', fontsize=8)

    plt.savefig(path, dpi=300)


# Data Manipulation
"""
A função verifica a porcentagem de valores faltantes em cada coluna e salva as informações
em um arquivo. Se ocorrer algum erro durante essa verificação, uma mensagem de erro será exibida.
"""

def valores_miss_nulls_cols(df: pd.DataFrame, output_path: str, col_type: str):


    try:
        if col_type == 'nulo':
            # Encontrar colunas com valores ausentes
            zero_cols = []
            for col in df.columns:
                zero_perc = df[col].isna().sum() / len(df) * 100
                if zero_perc > 0:
                    zero_cols.append((col, zero_perc))
                    
            # Salva as colunas com valores nulos e a porcentagem
            with open(output_path, 'w') as file:
                for col, zero_perc in zero_cols:
                    file.write(f"{col} ({zero_perc:.2f}% nulos)\n")

        elif col_type == 'faltante':
            # Encontrar colunas com valores ausentes
            miss_cols = []
            for col in df.columns:
                miss_perc = df[col].isna().sum() / len(df) * 100
                if miss_perc > 0:
                    miss_cols.append((col, miss_perc))
                    
            # Salva as colunas com valores nulos e a porcentagem
            with open(output_path, 'w') as file:
                for col, miss_perc in miss_cols:
                    file.write(f"{col} ({miss_perc:.2f}% faltantes)\n")
    
    except Exception as e:
        # Captura qualquer exceção que ocorrer durante a execução do código
        print(f"Ocorreu um erro: {e}")

"""
Remove colunas de um DataFrame que possuem uma porcentagem alta de dados faltantes,
salva as colunas excluídas em um arquivo de texto e retorna o DataFrame modificado.
"""
def drop_cols_high_miss_perc(df, output_path:str, threshold=float(50)):


    try:
        # Calcula a porcentagem de dados faltantes para cada coluna
        missing_percentages = df.isna().mean() * 100

        # Encontra as colunas que excedem o limite de porcentagem de dados faltantes
        cols_drop = missing_percentages[missing_percentages > threshold].index

        # Salva em texto as colunas excluídas
        with open(output_path, 'w') as file:
            for col in cols_drop:
                file.write(col + '\n')

        # Remove as colunas do DataFrame
        df = df.drop(columns=cols_drop)

        return df

    except KeyError:
        print("Algumas colunas já foram excluídas anteriormente.")
        return df
   
    except Exception as erro:
       print(erro)