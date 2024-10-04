from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd


print("Loading dataset")
dataset_filepath = "data/nutrition.csv"
df = pd.read_csv(dataset_filepath, index_col=0, header=[0, 1])
cols_to_keep = pd.MultiIndex.from_product(
    [["macro"], ["calories", "total_fat", "protein", "carbohydrate", "water"]]
)
df = df.loc[:, [pd.IndexSlice["identifier", "name"], *cols_to_keep]]


print(f"... Dataset loaded: {df.shape}")


app = Dash()

data_columns = df.columns.get_level_values(-1)
df.columns = data_columns

app.layout = [
    html.H1(children="Dataset Exploration", style={"textAlign": "center"}),
    dcc.Dropdown(data_columns, id="dropdown-1"),
    dcc.Dropdown(data_columns, id="dropdown-2"),
    dcc.Dropdown(data_columns, id="dropdown-3"),
    dcc.Graph(id="graph-content"),
]


@callback(
    Output("graph-content", "figure"),
    Input("dropdown-1", "value"),
    Input("dropdown-2", "value"),
    Input("dropdown-3", "value"),
)
def update_graph(var1, var2, var3):

    var_bools = [v is not None for v in [var1, var2, var3]]

    fig = {}

    if var_bools == [True, False, False]:
        print("return histogram")
        fig = px.histogram(df[var1], x=var1)

    if var_bools == [True, True, False]:
        print("return scatter 2D")
        fig = px.scatter(df, x=var1, y=var2)

    if var_bools == [True, True, True]:
        print("return scatter 3D")
        fig = px.scatter_3d(df, x=var1, y=var2, z=var3)

    if fig is not None:
        return fig


if __name__ == "__main__":
    app.run(debug=True)
