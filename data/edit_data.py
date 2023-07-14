import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

"""
A função edit_dataframe realiza a edição e transformação de um DataFrame, incluindo o mapeamento de colunas, 
limpeza de dados, geração de gráficos e modificação de valores. Ela carrega um arquivo CSV contendo os dados, 
renomeia as colunas usando um arquivo JSON de mapeamento, realiza ajustes nas colunas 'idade' e 'genero', 
gera um gráfico de distribuição de idade por gênero e modifica os valores de várias colunas para melhor 
interpretação. O DataFrame resultante é salvo em um novo arquivo CSV. Essa função simplifica o processo de 
preparação de dados, tornando-o mais eficiente e organizado.
"""
def edit_dataframe():

    # Carrega o arquivo .csv na pasta
    def load_file():
        
        try:
            directory = 'data/preprocessed_files/all_years/'
            file_list = [file for file in os.listdir(directory) if file.endswith('.csv')]
            dfs = []

            for file in file_list:
                file_path = os.path.join(directory, file)
                df = pd.read_csv(file_path)
                dfs.append(df)

            if dfs:
                df_concat = pd.concat(dfs, ignore_index=True)
                return df_concat
            else:
                print('Nenhum arquivo CSV encontrado na pasta.')
                return None
        except Exception as e:
            print(f'Erro ao carregar os arquivos CSV: {e}')
            return None

    # Renomeia as colunas
    def columns_mapping(df: pd.DataFrame):
        columns_mapping = {}
        # Nicknanes paras as colunas em arquivo json
        with open('data/preprocessed_files/all_years/json_files/variable_mapping.json', 'r') as file:
            columns_mapping = json.load(file)

        # Renomeia o nome das colunas para nomes mais curtos
        df.rename(columns=columns_mapping, inplace=True)

        return df

    # Trabalha com a variáveis idade e gênero 
    def age_gender_featuring(df: pd.DataFrame):
        
        # Ajeitando a idade
        try:
            # Converte todos os valores para numéricos e substitui os 'erros' para nan
            df['idade'] = pd.to_numeric(df['idade'], errors='coerce')

            condition = (df['idade'] < 18) | (df['idade'] > 100)
            df['idade'] = np.where(condition, np.nan, df['idade'])

        except KeyError:
            print(f"Erro: A coluna 'idade' não existe no DataFrame.")
        
        except ValueError:
            print(f"Erro: Dados inválidos na coluna idade")

        # Ajeitando o genero
        try:    
            gender_mapping = {}
            with open('data/preprocessed_files/all_years/json_files/gender_mapping.json', 'r') as file:
                gender_mapping = json.load(file)

            df['genero'] = df['genero'].astype(str)
            df['genero'] = df['genero'].str.replace(r'\d+', '')
            df['genero'] = df['genero'].str.lower().map(gender_mapping).fillna('Other')
        
        except KeyError:
            print(f"Erro: A coluna genero não existe no DataFrame.")
        
        except ValueError:
            print(f"Erro: Dados inválidos na coluna gender")

        
        # Gráfico
        df_ = df.dropna(subset=['idade'])

        sns.set(style='white', font_scale=1.0)

        # Parameters
        title = 'Dist. Idade por Gênero'
        x_axis = 'idade'
        x_label = 'Idade'
        hue = 'genero'
        
        custom_pallete = {'M': 'indigo', 'F': 'limegreen', 'Other': 'thistle'}
        sns.histplot(data=df_, x=x_axis, hue=hue, kde=True, palette=custom_pallete)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('Contagem')

        plt.close()

        os.makedirs('images', exist_ok=True)
        save_path = 'images/dist_idade_sexo.png'
        plt.savefig(save_path, dpi=400)

        return df

    # Edita as variáveis para um melhor encode
    def modify_dataframe(df: pd.DataFrame):

        # Edição dos valores para melhor interpretação das colunas

        # Alguns padrões que se repetem
        dict_1 = {
            '1.0': 'Yes',
            'True': 'Yes', 
            'False': 'No',
            '0.0': 'No'
            }

        # Colunas
        df['emp_tec'] = df['emp_tec'].map(dict_1)

        df['papel_tec'] = df['papel_tec'].map(dict_1)

        df['discussao_saude_mental_emp'] = df['discussao_saude_mental_emp'].map(dict_1)

        
        index_rename = {'Not eligible for coverage / NA': 'Not eligible'}
        df.replace({'beneficios_mental': index_rename}, inplace=True)
        
        df['conforto_discutir_colegas'] = df['conforto_discutir_colegas'].map({
        'Maybe': 'Maybe',
        'Yes': 'Yes', 
        'No': 'No'
        })    

        df['discussao_saude_mental_colegas'] = df['discussao_saude_mental_colegas'].map({
        '1': 'Yes', 
        '0': 'No',
        'No': 'No',
        'Yes': 'Yes',
        'True': 'Yes',
        'False': 'No',
        '0.0': 'No',
        '1.0': 'Yes'
        })

        df['discussao_saude_mental_colegas_outro'] = df['discussao_saude_mental_colegas_outro'].map({
        '1': 'Yes', 
        '0': 'No',
        'True': 'Yes',
        'False': 'No',
        '0.0': 'No',
        '1.0': 'Yes'
        })

        df['importancia_saude_fisica_emp'] = pd.to_numeric(df['importancia_saude_fisica_emp'], errors='coerce')

        df['importancia_saude_mental_emp'] = pd.to_numeric(df['importancia_saude_mental_emp'], errors='coerce')

        df['emp_tec_anterior'] = df['emp_tec_anterior'].map({'True': 1, 'False': 0})
        df['emp_tec_anterior'] = pd.to_numeric(df['emp_tec_anterior'], errors='coerce')

        # Map the values according to the specifications
        df['disturbio_saude_mental_atual'] = df['disturbio_saude_mental_atual'].map({
            "Don't Know": "I don't know",
            '0': 'No',
            'No': 'No',
            '1': 'Yes',
            '4': 'Yes',
            '7': 'Yes',
            '5': 'Yes',
            '8': 'Yes',
            '2': 'Yes',
            '3': 'Yes',
            'Yes': 'Yes',
            'Possibly': 'Maybe'
        })

        df['tratamento_profissional_mental'] = df['tratamento_profissional_mental'].map({
        "Don't Know": "I don't know",
        '0': 'No',
        'No': 'No',
        'FALSE': 'No',
        '1': 'Yes',
        'TRUE': 'Yes',
        'Yes': 'Yes',
        'Possibly': 'Maybe'
        })

        df['historico_familiar_mental'] = df['historico_familiar_mental'].map({
        "I don't Know": "I don't know",
        '0': 'No',
        'No': 'No',
        'FALSE': 'No',
        '1': 'Yes',
        'TRUE': 'Yes',
        'Yes': 'Yes',
        'Possibly': 'Maybe'
        })


        df['interferencia_sem_tratamento_eficaz'] = df['interferencia_sem_tratamento_eficaz'].map(
        {"1":"Yes",
        "Sometimes":"Sometimes",
        "Often": "Often",
        "Not applicable to me": "Not applicable to me",
        "Rarely": "Rarely",
        "Never": "Never",
        })

        # Map the values according to the specifications
        df['disposto_compartilhar_amigos_familiares'] = df['disposto_compartilhar_amigos_familiares'].map({
            "Sometimes": "Sometimes",
            "Maybe": "Maybe",
            '0': 'No',
            'No': 'No',
            '1': 'Yes',
            '5': 'Yes',
            '4': 'Yes',
            '10': 'Yes',
            '8': 'Yes',
            '3': 'Yes',
            '2': 'Yes',
            '9': 'Yes',
            '7': 'Yes',
            '1': 'Yes',
            'Yes': 'Yes',
        })

        # Map the values according to the specifications
        df['disposto_apresentar_saude_entrevista'] = df['disposto_apresentar_saude_entrevista'].map({
            "Yes": "Yes",
            "Maybe": "Maybe",
            '0': 'No',
            'No': 'No',
            '8': 'Yes',
            '7': 'Yes',
            '3': 'Yes',
            '10': 'Yes',
            '6': 'Yes',
            '1': 'Yes',
            '5': 'Yes',
            '9': 'Yes',
            'Often': 'Often'
        })

        df['identificado_abertamente_mental'] = df['identificado_abertamente_mental'].map({
        'TRUE': 'Yes',
        'FALSE': 'No',
        '1': 'Yes',
        '0': 'No'
        })

        df['traria_saude_mental_entrevista'] = df['traria_saude_mental_entrevista'].map({
        'No': 'No',
        'Maybe': 'Maybe',
        'Yes': 'Yes'
        })

        return df
    
    # Reservar o texto para análise NLP
    def save_text_dataframe(df):
        try:
        # Reservando o texto
            df_ = df.copy()    
            coluns_nlp = ['sugestoes_apoio_mental']
            df_ = df[coluns_nlp]
            output_file = 'data/nlp/mental_health_text.csv'
            df_.to_csv(output_file, index=False)
            return None
        
        except Exception as e:
            # Tratamento de erros
            print(f"Erro ao salvar o dataframe de texto: {str(e)}")
            raise
    
    # Chama as funções
    df = load_file()
    df = columns_mapping(df)
    save_text_dataframe(df=df)
    df = age_gender_featuring(df)
    df = modify_dataframe(df)
    
    # Salva o datafreame modificado
    os.makedirs('data/data_ready_to_model', exist_ok=True)
    save_path = 'data/data_ready_to_model/mental_health.csv'
    df.to_csv(save_path, index=False)

    print('Edição dos dados concluída.')

    