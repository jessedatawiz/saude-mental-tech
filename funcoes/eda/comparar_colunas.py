import pandas as pd

def comparar_colunas(df1, df2):
    colunas_diferentes = []

    # Verifica as colunas do dataframe1 que não estão presentes no dataframe2
    for coluna in df1.columns:
        if coluna not in df2.columns:
            colunas_diferentes.append((coluna, "df1"))

    # Verifica as colunas do dataframe2 que não estão presentes no dataframe1
    for coluna in df2.columns:
        if coluna not in df1.columns:
            colunas_diferentes.append((coluna, "df2"))

    return colunas_diferentes