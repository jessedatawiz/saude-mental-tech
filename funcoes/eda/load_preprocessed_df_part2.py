import pandas as pd

def load_preprocessed_df_part2(ano):
    try:
        
        file_path = f"./arquivos/pre_process/eda_part2/{ano}.csv"
        df_pre = pd.read_csv(file_path)
        return df_pre
    except FileNotFoundError:
        print(f"Arquivo não encontrado, {ano}")