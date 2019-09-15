import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

CORES = mp.cpu_count()-2

def parallelize_df(df, partitions, func):
    df_split = np.array_split(df, partitions)
    pool = mp.Pool(CORES)
    df = pd.concat(pool.map(func,df_split))
    pool.close()
    pool.join()
    return df

def parallelize_df_np(df, partitions, func):
    df_split = np.array_split(df, partitions)
    pool = mp.Pool(1)
    result = np.concatenate(pool.map(func,df_split))
    pool.close()
    pool.join()
    return result()

