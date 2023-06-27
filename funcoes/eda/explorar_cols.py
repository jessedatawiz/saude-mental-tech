import pandas as pd

def explorar_cols(df: pd.DataFrame, col_index: list):

    col_index = col_index
    col_nomes = [f"{df.columns[i]} ({df.iloc[:, i].isna().mean() * 100:.2f}% missing)" for i in col_index]
    for i in col_nomes:
        print(i)