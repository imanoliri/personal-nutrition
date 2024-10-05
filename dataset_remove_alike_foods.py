from alike_foods import (
    get_nutrition_dataset_metadata,
    report_nutrition_dataset_metadata,
)
import pandas as pd

dataset_filepath = "data/nutrition.csv"
print(f"Loading: {dataset_filepath}")
df = pd.read_csv(dataset_filepath, index_col=0, header=[0, 1])
print(f"Loaded: {df.shape}")


def reduce_dataset_in_batches_iteratively(
    df: pd.DataFrame, batch_size: int = 350
) -> pd.DataFrame:

    if len(df) <= batch_size * 3:
        return reduce_dataset(df, return_metadata=True)
    df_reduced = pre_reduce_dataset_in_batches(df, batch_size)
    if len(df_reduced) == len(df):
        return reduce_dataset(df, return_metadata=True)
    return reduce_dataset_in_batches_iteratively(df_reduced, batch_size)


def pre_reduce_dataset_in_batches(
    df: pd.DataFrame, batch_size: int = 350
) -> pd.DataFrame:

    dfs_reduced = []
    print(f"\n---> Reducing dataset size {df.shape} in batches of size {batch_size}")

    idx_delimiters = [*range(0, len(df), batch_size), len(df)]
    for b, (id_s, id_e) in enumerate(zip(idx_delimiters[:-1], idx_delimiters[1:])):
        print(f"Batch {b+1}")
        dfs_reduced.append(reduce_dataset(df.iloc[id_s:id_e]))

    return pd.concat(dfs_reduced, axis=0)


def reduce_dataset(df: pd.DataFrame, return_metadata: bool = False) -> pd.DataFrame:
    df_metadata = get_nutrition_dataset_metadata(df)
    ids_to_remove = df_metadata[-1]
    df_reduced = df.loc[~df.index.isin(ids_to_remove)]
    if return_metadata:
        return df_reduced, df_metadata
    return df_reduced


df_reduced, df_reduced_metadata = reduce_dataset_in_batches_iteratively(df, 50)

print(f"\nReduced to {df_reduced.shape}")

report_nutrition_dataset_metadata("results/dataset_metadata", df, *df_reduced_metadata)
