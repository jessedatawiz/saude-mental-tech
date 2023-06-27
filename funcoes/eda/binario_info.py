import pandas as pd

def binario_info(df: pd.DataFrame, number_col: int):
        
    """
    Calcula e exibe a porcentagem de 0s e 1s em uma coluna binária.

    Args:
        df (pd.DataFrame): O DataFrame contendo a coluna binária.
        number_col (int): O índice da coluna binária.

    Returns:
        None
    """
    vc = df.iloc[:, number_col].value_counts()

    total = vc.sum()

    percentage_0 = (vc[0] / total) * 100
    percentage_1 = (vc[1] / total) * 100

    print(f"Porcentagem de NÃO: {percentage_0:.2f}%")
    print(f"Porcentagem de SIM: {percentage_1:.2f}%")

    return None