import numpy as np
import pandas as pd

def miss_zero_cols(df: pd.DataFrame, arquivo_out: str, col_type: str):


    try:
        if col_type == 'zero':
            # Encontrar colunas com valores ausentes
            zero_cols = []
            for col in df.columns:
                zero_perc = df[col].isna().sum() / len(df) * 100
                if zero_perc > 0:
                    zero_cols.append((col, zero_perc))
                    
            # Salva as colunas com valores nulos e a porcentagem
            with open(arquivo_out, 'w') as file:
                for col, zero_perc in zero_cols:
                    file.write(f"{col} ({zero_perc:.2f}% nulos)\n")

        elif col_type == 'miss':
            # Encontrar colunas com valores ausentes
            miss_cols = []
            for col in df.columns:
                miss_perc = df[col].isna().sum() / len(df) * 100
                if miss_perc > 0:
                    miss_cols.append((col, miss_perc))
                    
            # Salva as colunas com valores nulos e a porcentagem
            with open(arquivo_out, 'w') as file:
                for col, miss_perc in miss_cols:
                    file.write(f"{col} ({miss_perc:.2f}% faltantes)\n")
    
    except Exception as e:
        # Captura qualquer exceção que ocorrer durante a execução do código
        print(f"Ocorreu um erro: {e}")

