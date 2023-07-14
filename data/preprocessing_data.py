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

A função retorna o DataFrame processado.
"""
import os
import sys
import pandas as pd

# Adicione o diretório pai ao sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.utils import translate, drop_cols_high_miss_perc

def call_preprocess_data():
    # Função principal para pré-processamento
    def preprocess_data(ano) -> pd.DataFrame:

        
        # Ler o arquivo CSV
        try:   
            arquivo_entrada = f'./mental_files/osmi_smt_{ano}.csv'
            df = pd.read_csv(arquivo_entrada)
        except FileNotFoundError:
            print(f"O arquivo {arquivo_entrada} não foi encontrado.")
        except Exception as e:
            print(f"Ocorreu um erro ao ler o arquivo CSV: {arquivo_entrada}, {e}")

        # Limpar os nomes das colunas das tags HTML
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
            path_cols_droped = f'./data/eda/{ano}/colunas_del_50_miss.txt'
            dir_out_cols = os.path.dirname(path_cols_droped)
            os.makedirs(dir_out_cols, exist_ok=True)
            df = drop_cols_high_miss_perc(df, path_cols_droped)
        except FileNotFoundError:
            print(f"O arquivo {path_cols_droped} não foi encontrado.")
        except Exception as e:
            print(f"Ocorreu um erro ao remover as colunas com valores ausentes: {e}")


        # Traduzir os nomes das colunas para o português
        try:
            df = translate(df, destination_language='pt')
        except Exception as e:
            print(f"Ocorreu um erro ao traduzir os nomes das colunas: {e}")

        # Remover colunas específicas que contêm palavras-chave
        try:
            cols_descartar = df.columns.str.contains("EUA|Por que|anterior|anteriores|país|UTC|raça|ID")
            df.drop(columns=df.columns[cols_descartar], inplace=True)
            if ano == 2017 or ano == 2018:
                coluna_excluir = "Você estaria disposto a conversar com um de nós mais extensivamente sobre suas experiências com problemas de saúde mental no setor de tecnologia?(Observe que todas as respostas da entrevista seriam usadas anonimamente e apenas com sua permissão.)"
            else:
                coluna_excluir = "Você estaria disposto a conversar com um de nós mais extensivamente sobre suas experiências com problemas de saúde mental no setor de tecnologia?(Observe que todas as respostas da entrevista seriam usadas _Anonymly_ e somente com sua permissão.)"
            df.drop(columns=coluna_excluir, inplace=True)
        except Exception as e:
            print(f"Ocorreu um erro ao remover as colunas específicas: {e}")

        # Salva df pré-processado
        try:
            path_df_pre = f"./data/preprocessed_files/{ano}/{ano}.csv"
            dir_df_pre = os.path.dirname(path_df_pre)
            os.makedirs(dir_df_pre, exist_ok=True)
            df.to_csv(path_df_pre, index=False)
            
        except Exception as e:
            print("Ocorreu um erro ao salvar o dataframe intermediário.")
            print("Erro: ", str(e))

        print(f'Pré-processamento concluído -> {ano}.')
        return None

    # Executa a primeira parte do pré-processamento
    for year in range(2017, 2022):
        preprocess_data(year)


"""
A parte 2 deste arquivo contém uma função chamada process_data() que realiza o processamento de dados relacionados à saúde mental de funcionários em diferentes anos.
A função renomeia colunas específicas nos dois primeiros dataframes da lista dfs.
É adicionada uma coluna 'Ano' em cada dataframe, indicando o ano correspondente.
Os dataframes são concatenados em um único dataframe.
São gerados gráficos relacionados ao número de participantes e número de colunas dos dataframes.
São salvos os valores nulos/faltantes em arquivos separados.
O dataframe unificado é salvo em um arquivo CSV.
O texto de uma coluna específica é reservado para análise NLP e salvo em um arquivo separado.
A função trata erros e exibe mensagens de erro, caso ocorram.
Ao final do processamento, é exibida uma mensagem de conclusão bem-sucedida.
"""
# Adicione o diretório pai ao sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.utils import load_csv_files, plot_num_cols, plot_num_participantes, valores_miss_nulls_cols

def rename_columns(dfs):
    try:
        cols_rename = {
            "Se você tem um distúrbio de saúde mental, com que frequência sente que isso interfere no seu trabalho quando não é tratado de maneira eficaz (ou seja, quando está experimentando sintomas)?":
            "Se você tem um distúrbio de saúde mental, com que frequência sente que isso interfere no seu trabalho quando não é tratado de maneira eficaz (ou seja, quando você está experimentando sintomas)?",
            "Se eles soubessem que você sofria de um distúrbio de saúde mental, como você acha que os membros da equipe/colegas de trabalho reagiriam?":
            "Se eles soubessem que você sofria de um distúrbio de saúde mental, como você acha que os membros da sua equipe/colegas de trabalho reagiriam?"
        }

        for df in dfs[:2]:  # Apenas para os dois primeiros dataframes (2017 e 2018)
            df.rename(columns=cols_rename, inplace=True)
        return dfs

    except Exception as e:
        # Tratamento de erros
        print(f"Erro ao renomear colunas: {str(e)}")
        raise

def add_year_column(dfs):
    try:
        for i, df in enumerate(dfs):
            year = 2017 + i
            df['Ano'] = year
            df['Ano'] = pd.to_datetime(df['Ano'], format='%Y')
        
        return dfs

    except Exception as e:
        # Tratamento de erros
        print(f"Erro ao adicionar coluna de ano: {str(e)}")
        raise

def save_merged_dataframe(df):
    try:
        df_copy = df.copy()

        # Exclui a coluna destinada para nlp
        # columns_nlp = ['Descreva brevemente o que você acha que o setor como um todo e/ou empregadores poderia fazer para melhorar o apoio à saúde mental aos funcionários.']
        # df_copy = df_copy.drop(columns_nlp, axis=1)
        output_file = 'data/preprocessed_files/all_years/mental_health_precleaned.csv'
        df_copy.to_csv(output_file, index=False) 
        return None

    except Exception as e:
        # Tratamento de erros
        print(f"Erro ao salvar o dataframe mesclado: {str(e)}")
        raise

def process_data():
    try:
        # Carrega os dataframes já pré-processados na parte 1
        file_paths = ['data/preprocessed_files/2017/2017.csv',
                      'data/preprocessed_files/2018/2018.csv',
                      'data/preprocessed_files/2019/2019.csv',
                      'data/preprocessed_files/2020/2020.csv',
                      'data/preprocessed_files/2021/2021.csv']
        dfs = load_csv_files(file_paths)

        # Renomear as colunas
        dfs = rename_columns(dfs=dfs)
        
        # Salvar as informações gerais das colunas recém-padronizadas
        os.makedirs('data/preprocessed_files/all_years/images', exist_ok=True)

        output_path = 'data/preprocessed_files/all_years/images/numero_participantes.png'
        plot_num_participantes(dfs, output_path)

        output_path = 'data/preprocessed_files/all_years/images/numero_colunas.png'
        plot_num_cols(dfs, output_path)

        # Adicionar coluna 'Ano'
        dfs = add_year_column(dfs=dfs)

        # Concatenando os dataframes em um só
        merged_df = pd.concat(dfs, ignore_index=True)

        # Salvar a porncentagem de valores faltantes/nulos no dataframe unificado
        output_path = 'data/preprocessed_files/all_years/valores_nulos.txt'
        valores_miss_nulls_cols(merged_df, output_path, 'nulo')
        output_path = 'data/preprocessed_files/all_years/valores_faltantes.txt'
        valores_miss_nulls_cols(merged_df, output_path, 'faltante')

        # Salvar o novo dataframe unificado, sem a coluna do nlp
        save_merged_dataframe(df=merged_df)

    except Exception as e:
        # Tratamento de erros
        print(f"Ocorreu um erro durante o processamento dos dados: {str(e)}")
        return None

    print("Processamento, parte 2, dos dados concluído.")
