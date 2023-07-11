import pandas as pd

def load_csv_files(file_paths):
    dataframes = []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    return dataframes


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

