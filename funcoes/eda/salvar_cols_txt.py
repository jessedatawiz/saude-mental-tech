import pandas as pd


def salvar_colunas_em_txt(lista_df, nome_arquivo):
    
    
    anos = list(range(2016,2022))
    try:
        with open(nome_arquivo, 'w') as arquivo:
            for df,ano in zip(lista_df, anos):
                colunas = df.columns.to_list()
                arquivo.write(str(ano))
                arquivo.write('\n')
                
                for coluna in colunas:
                    arquivo.write(str(coluna))
                    arquivo.write('\n')
                arquivo.write('\n')

        print(f"As colunas dos DataFrames foram salvas no arquivo {nome_arquivo} com sucesso!")

    except FileNotFoundError:
        print(f"Erro: o arquivo ou diretório não foi encontrado.")
    except PermissionError:
        print(f"Erro: permissão negada para escrever o arquivo.")
    except Exception as e:
        print(f"Ocorreu um erro ao salvar as colunas no arquivo.")
        print("Erro: ", str(e))


