import pandas as pd
import plotly.express as px
from typing import List, Tuple
from sklearn.decomposition import PCA


def get_macro_micro_full_datasets(df):
    macro_core_cols = ["calories", "carbohydrate", "protein", "fat", "water"]
    macro_cols = df.loc[:, pd.IndexSlice["macro", :]].columns
    micro_cols = df.loc[:, pd.IndexSlice["micro", :]].columns
    cols_macro_core = pd.MultiIndex.from_product([["macro"], macro_core_cols])
    cols_macro_micro = pd.MultiIndex.from_tuples(
        [
            *macro_cols,
            *micro_cols,
        ]
    )
    df_ma = df.loc[:, [pd.IndexSlice["identifier", "name"], *cols_macro_core]].dropna()
    df_mi = df.loc[:, [pd.IndexSlice["identifier", "name"], *micro_cols]].dropna()
    df_f = df.loc[:, [pd.IndexSlice["identifier", "name"], *cols_macro_micro]].dropna()
    df_ma.columns = df_ma.columns.droplevel(0)
    df_mi.columns = df_mi.columns.droplevel(0)
    df_f.columns = df_f.columns.droplevel(0)
    return df_ma, df_mi, df_f


def get_pca(df: pd.DataFrame, n_components=5) -> Tuple[PCA, pd.DataFrame, List[str]]:
    df_v = df.iloc[:, 1:]
    pca = PCA(n_components=min(n_components, df_v.shape[1]))
    pca.fit(df_v)
    # df_r_v_t = pca_r.transform(df_r_v)
    return get_pca_fig(pca)


def get_pca_fig(pca: PCA, title: str = None):

    comps = pca_component_matrix(pca)

    fig = px.imshow(
        comps.drop(columns=["explained_variance", "comp"]),
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title,
        text_auto=True,
    )

    autoreport = "\n".join(pca_autoreport(pca))

    return fig, comps, autoreport


def pca_component_matrix(pca: PCA) -> pd.DataFrame:
    columns = pca.feature_names_in_
    comps = pca.components_
    comps = pd.DataFrame(comps.round(2), columns=columns)
    comps = comps.loc[:, columns]
    comps.insert(0, "comp", [f"Comp_{i+1}" for i in range(comps.shape[0])])
    comps.insert(
        0, "explained_variance", (100 * pca.explained_variance_ratio_).round(0)
    )
    return comps


def pca_autoreport(pca: PCA, top_comps: int = 5, top_factors=5) -> List[str]:
    autoreport_lines = []
    autoreport_lines.append("Top components are:\n")

    comps = pca_component_matrix(pca).iloc[:, 2:]
    comps_top = comps.iloc[:top_comps]
    top_comps_and_factors = [
        comp.loc[comp.abs().sort_values(ascending=False).iloc[:top_factors].index]
        for r, comp in comps_top.iterrows()
    ]
    lines_top_comps = [
        "; ".join(f"{k}: {v}" for k, v in cf.to_dict().items())
        for cf in top_comps_and_factors
    ]
    autoreport_lines += [
        f"{c} --> {nl}" for c, nl in zip(comps.index[:top_comps], lines_top_comps)
    ]
    return autoreport_lines
