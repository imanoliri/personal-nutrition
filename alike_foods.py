import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path

# python -m cProfile -o alike_foods_report.prof backlog/alike_foods.py > alike_foods_report.txt
# snakeviz alike_foods_report.prof


def get_unique_food_identifiers(df: pd.DataFrame) -> List[str]:
    return list(
        set(
            [
                v
                for values in df.identifier.iloc[:, 1:].values
                for v in values
                if not pd.isnull(v)
            ]
        )
    )


def compare_data(df: pd.DataFrame) -> pd.DataFrame:
    ids1 = np.array([], dtype=int)
    ids2 = np.array([], dtype=int)
    name_likeness = np.array([], dtype=float)
    macro_differences = np.array([], dtype=float)

    macro_cols = ["total_fat", "protein", "carbohydrate"]

    for f1 in df.index:
        for f2 in df.index:
            if f1 == f2:
                continue
            if sum((ids1 == f2) & (ids2 == f1)) > 0:
                continue

            # ids
            ids1 = np.append(ids1, f1)
            ids2 = np.append(ids2, f2)

            # Name likeness
            dfi = df.identifier
            comparing_tags = [
                (v1, v2)
                for v1, v2 in zip(dfi.loc[f1].iloc[1:], dfi.loc[f2].iloc[1:])
                if not pd.isnull(v1) or not pd.isnull(v2)
            ]
            same_tags = [v1 == v2 for v1, v2 in comparing_tags]
            like = sum(same_tags) / len(comparing_tags)
            name_likeness = np.append(name_likeness, like)

            # Macro difference
            dfm = df.macro.loc[:, macro_cols]
            macro_difference = (dfm.loc[f1] - dfm.loc[f2]).abs().sum()
            macro_differences = np.append(macro_differences, macro_difference)

    df_comparison = pd.DataFrame(
        [ids1, ids2, name_likeness, macro_differences],
        index=["id1", "id2", "name_likeness", "macro_difference"],
    ).T
    df_comparison["id1"] = df_comparison["id1"].astype(int)
    df_comparison["id2"] = df_comparison["id2"].astype(int)
    return df_comparison


def filter_alike_food_pairs(
    dfc: pd.DataFrame,
    consider_name_likeness: bool = True,
    consider_macro_difference: bool = True,
    name_likeness_threshold: float = 0.7,
    macro_difference_threshold: float = 5,
) -> pd.DataFrame:
    if not consider_name_likeness and not consider_macro_difference:
        return dfc

    likeness_mask = None
    if consider_name_likeness:
        new_mask = dfc.name_likeness > name_likeness_threshold
        if likeness_mask is None:
            likeness_mask = new_mask
        else:
            likeness_mask = likeness_mask & new_mask
    if consider_macro_difference:
        new_mask = dfc.macro_difference < macro_difference_threshold
        if likeness_mask is None:
            likeness_mask = new_mask
        else:
            likeness_mask = likeness_mask & new_mask
    return (
        dfc.loc[likeness_mask]
        .sort_values("name_likeness", ascending=False)
        .sort_values("macro_difference")
    )


def groups_from_pairs(pairs: List[Tuple[int]]) -> List[List[int]]:
    alike_groups = []
    for p1 in pairs:
        # If it belongs to a previous alike_group, continue the old group, else start a new one
        new_group = True
        alike_group = np.array(p1, dtype=int)
        for agroup in alike_groups:
            if any(idp1 in agroup for idp1 in p1):
                new_group = False
                alike_group = agroup
                continue

        # Add other pairs to the group
        for p2 in pairs:
            if (p1 == p2).all():
                continue

            if any(idp1 in p2 for idp1 in p1):
                alike_group = np.append(alike_group, p2)

        # If it's a new group, add it to the list!
        if new_group:
            alike_groups.append(np.unique(alike_group))

    return alike_groups


def get_nutrition_dataset_metadata(
    df: pd.DataFrame,
    consider_name_likeness: bool = False,
    consider_macro_difference: bool = True,
    name_likeness_threshold: float = 0.7,
    macro_difference_threshold: float = 5,
):

    # Reduce data columns
    cols_to_keep = pd.MultiIndex.from_product([["identifier"], df.identifier.columns])
    cols_to_keep = cols_to_keep.append(
        pd.MultiIndex.from_product(
            [["macro"], ["total_fat", "protein", "carbohydrate"]]
        )
    )
    df = df.loc[:, cols_to_keep].copy(deep=True)

    # Unique labels in comparison to food items
    unique_identifiers = get_unique_food_identifiers(df)

    # Get alike food groups
    dfc = compare_data(df)

    dfs = filter_alike_food_pairs(
        dfc,
        consider_name_likeness,
        consider_macro_difference,
        name_likeness_threshold,
        macro_difference_threshold,
    )

    alike_groups = groups_from_pairs(dfs.iloc[:, :2].values)

    return (unique_identifiers, dfc, dfs, alike_groups)


def report_nutrition_dataset_metadata(
    path: str,
    df: pd.DataFrame,
    unique_identifiers: List[int],
    dfc: pd.DataFrame,
    dfs: pd.DataFrame,
    alike_groups: List[np.ndarray],
    consider_name_likeness: bool = False,
    consider_macro_difference: bool = True,
    name_likeness_threshold: float = 0.7,
    macro_difference_threshold: float = 5,
):

    Path(path).mkdir(exist_ok=True, parents=True)

    # Conditions report str
    conditions_str = ""
    if consider_name_likeness:
        conditions_str += f"name likeness > {int(name_likeness_threshold*100)}%"
    if consider_macro_difference:
        conditions_str += (
            f"total macro difference of < {macro_difference_threshold} grams"
        )

    # Report
    metadata_report = []
    metadata_report.append(
        f"Unique food identifiers: {len(unique_identifiers)} for {len(df)} food items."
    )

    metadata_report.append(f"Food pair items: {len(dfc)}")

    metadata_report.append(f"Alike food pair items: {len(dfs)} (with {conditions_str})")
    df_most_alike_food_items = df.loc[dfs.iloc[0, :2].values]
    df_least_alike_food_items = df.loc[dfs.iloc[-1, :2].values]
    df_most_alike_food_items.to_csv(f"{path}/data_most_alike_food_items.csv")
    df_least_alike_food_items.to_csv(f"{path}/data_least_alike_food_items.csv")

    metadata_report.append(f"Alike food item groups: {len(alike_groups)}")
    alike_group_sizes = [len(ag) for ag in alike_groups]
    alike_foods = [
        [
            "("
            + ", ".join(
                str(int(round(p, 0)))
                for p in df.loc[ag[0]]
                .macro.loc[["total_fat", "protein", "carbohydrate"]]
                .tolist()
            )
            + ")",
            *df.loc[ag].identifier.name.tolist(),
        ]
        for ag in alike_groups
    ]

    metadata_report.append(
        f" (with a mean ammount of food items={str(int(np.mean(alike_group_sizes).round(0)))})"
    )

    def join_vals(vals):
        return "__".join(str(int(v)) for v in vals)

    agroup_hist_strs = [
        f"\t{join_vals(vals)}" for vals in np.histogram(alike_group_sizes)
    ]

    metadata_report.append("\n".join(agroup_hist_strs))

    pd.Series(alike_group_sizes).plot.hist(
        title=f"Food item group sizes with\n{conditions_str}"
    )
    plt.savefig(f"{path}/alike_food_group_sizes.jpg")
    plt.close()

    with open(f"{path}/metadata_report.txt", "w") as fp:
        fp.write("\n".join(metadata_report))

    with open(f"{path}/alike_food_ids.txt", "w") as fp:
        fp.write("\n".join(", ".join(str(i) for i in ag) for ag in alike_groups))

    with open(f"{path}/alike_foods.txt", "w") as fp:
        fp.write("\n".join("____".join(ag) for ag in alike_foods))
