import pandas as pd

def drop_linhas(df: pd.DataFrame, valor_ref: int, output_path: str):

    try:
        miss_perc = df.isna().mean() * 100

        linhas_drop = miss_perc[miss_perc > valor_ref].index

        # Salva as colunas excluídas em arquivo
        with open(output_path, 'w') as file:
            for col in linhas_drop:
                file.write(col + '\n')

        # Retorna o df com as linhas excluídas
        df = df.drop(labels=linhas_drop, axis=0)

        return df

    except KeyError:
        print("Algumas linhas já foram excluídas anteriormente.")
        return df