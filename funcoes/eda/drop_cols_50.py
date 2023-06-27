import pandas as pd

def drop_cols_50_perc(df, output_path:str, threshold=float(50)):


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