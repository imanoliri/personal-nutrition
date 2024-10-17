from pca import get_macro_micro_full_datasets, get_pca

import pandas as pd
import plotly.express as px


# Load data
dataset_filepath = "data/nutrition.csv"
results_dir = "results"

df = pd.read_csv(dataset_filepath, index_col=0, header=[0, 1])

# Separate datasets
df_partials = get_macro_micro_full_datasets(df)


pca_partials = []
for dfp in df_partials:
    pca_partials.append(get_pca(dfp))

partial_labels = ["macro", "micro", "full"]
pca_to_plot = dict(zip(partial_labels, pca_partials))
dataset_to_plot = dict(zip(partial_labels, df_partials))


# Web App
from dash import Dash, dcc, html, Input, Output, callback, dash_table


def render_PCA(label, fig_pca):
    return html.Div(
        [
            html.H3(label),
            dcc.Graph(figure=fig_pca[0]),
            dash_table.DataTable(
                fig_pca[1].to_dict("records"),
                [{"name": i, "id": i} for i in fig_pca[1].columns],
            ),
            dcc.Textarea(id="report", value=fig_pca[2], style=text_style),
        ]
    )


def render_data_exploration(label, fig_pca):
    data_columns_to_select = fig_pca[1].columns[2:]
    return html.Div(
        [
            html.H1(children=label, style={"textAlign": "center"}),
            dcc.Dropdown(data_columns_to_select, id="dropdown-1"),
            dcc.Dropdown(data_columns_to_select, id="dropdown-2"),
            dcc.Dropdown(data_columns_to_select, id="dropdown-3"),
            dcc.Graph(id="graph-content"),
        ]
    )


app = Dash(__name__)

df_plot = df.copy()
df_plot.columns = df_plot.columns.droplevel(0)


PCA_KEYWORD = "PCA"
DATASET_KEYWORD = "dataset exploration"
KEYWORDS = [PCA_KEYWORD, DATASET_KEYWORD]
RENDERING_FUNCTIONS = [render_PCA, render_data_exploration]
function_to_render = dict(zip(KEYWORDS, RENDERING_FUNCTIONS))

pca_labels = [f"{PCA_KEYWORD} {l}" for l in partial_labels]
data_labels = [f"{DATASET_KEYWORD} {l}" for l in partial_labels]
tab_values_labels = {
    f"tab-{i+1}": f"{k} {l}"
    for i, (k, l) in enumerate((k, l) for k in KEYWORDS for l in partial_labels)
}
tabs = [dcc.Tab(label=l, value=v) for v, l in tab_values_labels.items()]


app.layout = html.Div(
    [
        html.H1("PCA Tabs"),
        dcc.Tabs(
            id="tabs",
            value=tabs[0].value,
            children=tabs,
        ),
        html.Div(id="graph"),
    ]
)

text_style = {"width": 800, "height": 150}


@callback(Output("graph", "children"), Input("tabs", "value"))
def render_content(value):
    label = tab_values_labels[value]
    kw = [kw for kw in KEYWORDS if label.startswith(kw)][0]
    p = label[len(kw) :].strip()
    return function_to_render[kw](label, pca_to_plot[p])


@callback(
    Output("graph-content", "figure"),
    Input("dropdown-1", "value"),
    Input("dropdown-2", "value"),
    Input("dropdown-3", "value"),
)
def update_multigraph(var1, var2, var3):

    var_bools = [v is not None for v in [var1, var2, var3]]

    fig = {}

    if var_bools == [True, False, False]:
        fig = px.histogram(df_plot[var1], x=var1)

    if var_bools == [True, True, False]:
        fig = px.scatter(df_plot, x=var1, y=var2, hover_name="name")

    if var_bools == [True, True, True]:
        fig = px.scatter_3d(
            df_plot, x=var1, y=var2, z=var3, width=800, height=800, hover_name="name"
        )

    if fig is not None:
        return fig


if __name__ == "__main__":

    app.run(debug=True)
