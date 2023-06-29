import os
import pandas as pd

"""

A função preprocess_data é uma função que executa uma série de etapas para pré-processar
os dados de um arquivo CSV.

A função começa lendo o arquivo CSV especificado no parâmetro de entrada, exibindo uma 
mensagem de erro se o arquivo não for encontrado ou ocorrer algum erro durante a leitura.

Em seguida, a função limpa os nomes das colunas do DataFrame, removendo tags HTML. Se os 
nomes das colunas já estiverem limpos, uma mensagem informando isso será exibida.

A função, em seguida, remove colunas que possuem mais de 50% de valores ausentes, salvando 
as informações das colunas removidas em um arquivo. Se ocorrer algum erro durante a remoção 
das colunas, será exibida uma mensagem de erro.

Depois, a função traduz os nomes das colunas para o português, se necessário. Se ocorrer algum
erro durante a tradução, será exibida uma mensagem de erro.

A função também remove colunas específicas que contêm palavras-chave. Novamente, se ocorrer algum
erro durante essa etapa, será exibida uma mensagem de erro.

Por fim, a função verifica a porcentagem de valores faltantes em cada coluna e salva as informações
em um arquivo. Se ocorrer algum erro durante essa verificação, uma mensagem de erro será exibida.

A função retorna o DataFrame processado.

"""

# Tradução dos dados para português
from funcoes.eda.tradutor import traduz_cols

# Exploratory Data Analysis
from funcoes.eda.drop_cols_50 import drop_cols_50_perc
from funcoes.eda.miss_zero_values import miss_zero_cols


# Função principal para pré-processamento
def preprocess_data(ano) -> pd.DataFrame:

    
    # Ler o arquivo CSV
    try:   
        arquivo_entrada = f'./arquivos/dados_brutos/osmi_smt_{ano}.csv'
        df = pd.read_csv(arquivo_entrada)
    except FileNotFoundError:
        print(f"O arquivo {arquivo_entrada} não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo CSV: {e}")

    # Limpar os nomes das colunas
    try:
        df.columns = (df.columns.str.replace('<strong>', '')
                                .str.replace('</strong>', '')
                                .str.replace('<em>', '')
                                .str.replace('</em>', ''))
    except AttributeError:
        print("Os nomes das colunas já foram limpos.")
        pass
    except Exception as e:
        print(f"Ocorreu um erro ao limpar os nomes das colunas: {e}")

    # Remover colunas com mais de 50% de valores ausentes
    try:
        path_cols_droped = f'./arquivos/output/eda/{ano}/cols_droped.txt'
        dir_out_cols = os.path.dirname(path_cols_droped)
        os.makedirs(dir_out_cols, exist_ok=True)
        df = drop_cols_50_perc(df, path_cols_droped)
    except FileNotFoundError:
        print(f"O arquivo {path_cols_droped} não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao remover as colunas com valores ausentes: {e}")


    # Traduzir os nomes das colunas para o português (se necessário)
    try:
        df = traduz_cols(df, destination_language='pt')
    except Exception as e:
        print(f"Ocorreu um erro ao traduzir os nomes das colunas: {e}")

    # Remover colunas específicas que contêm palavras-chave
    try:
        cols_descartar = df.columns.str.contains('EUA|Por que|anterior|anteriores|país|UTC|raça|ID')
        df.drop(columns=df.columns[cols_descartar], inplace=True)
    except Exception as e:
        print(f"Ocorreu um erro ao remover as colunas específicas: {e}")


    # Salvar colunas com valores ausentes em um arquivo
    try:
        path_val_ausentes = f'./arquivos/output/eda/{ano}/val_ausentes.txt'
        dir_out_miss = os.path.dirname(path_val_ausentes)
        os.makedirs(dir_out_miss, exist_ok=True)

        # Verificar a porcentagem de valores faltantes
        miss_zero_cols(df, path_val_ausentes, 'miss')

    except FileNotFoundError:
        print(f"O arquivo {path_val_ausentes} não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro ao verificar a porcentagem de valores faltantes: {e}")


    return df