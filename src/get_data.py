import pandas as pd

def get_csv(local_path : str):
    return pd.read_csv(local_path, sep=",", low_memory=False,)