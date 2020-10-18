import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash.dependencies import ALL, Input, Output, State
from flask import Flask
from plotly.subplots import make_subplots
from scipy.stats import norm

param_dict = {
    "normal": ["mean", "variance"],
    "uniform": ["min", "max"],
    "beta": ["alpha", "beta"],
}

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Distribution Law"),
                dcc.Dropdown(
                    id="Distribution-Law",
                    options=[
                        {"label": col, "value": col}
                        for col in ["uniform", "normal", "beta"]
                    ],
                    value="uniform",
                ),
            ]
        ),
        dbc.FormGroup(id="Parameters"),
        dbc.FormGroup(
            [
                dbc.Col(
                    [
                        dbc.Row(dbc.Label("Population size (N):")),
                        dbc.Row(
                            dcc.Input(
                                id="n_simulation",
                                type="number",
                                debounce=True,
                                placeholder="N",
                                value=500,
                            )
                        ),
                        dbc.Row(
                            dbc.Label(id="text_n", children="Sample size (n): 100")
                        ),
                        dcc.Slider(
                            id="sample_size",
                            min=0,
                            max=4,
                            value=2,
                            step=0.01,
                            marks={0: "1", 1: "10", 2: "100", 3: "1000", 4: "10000"},
                        ),
                    ]
                ),
            ]
        ),
    ],
    body=True,
)

app.layout = dbc.Container(
    [
        dbc.Navbar([dbc.NavbarBrand("Central Limit Theorem", className="ml-1")]),
        dbc.Button(
            "Launch Simulations",
            color="primary",
            block=True,
            id="launch_simu",
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(controls, md=3),
                dbc.Col(dcc.Graph(id="CLT"), md=9),
            ],
            align="center",
        ),
    ],
    fluid=True,
)


@app.callback(Output("Parameters", "children"), Input("Distribution-Law", "value"))
def generate_param_box(distribution):
    param_list = param_dict[distribution]
    return [
        dcc.Input(
            id={"type": "param", "index": j},
            type="number",
            debounce=True,
            placeholder=param_list[j],
        )
        for j in range(len(param_list))
    ]


@app.callback(
    [Output("CLT", "figure"), Output("text_n", "children")],
    [
        Input("launch_simu", "n_clicks"),
        Input("n_simulation", "value"),
        Input("sample_size", "value"),
    ],
    [
        State("Distribution-Law", "value"),
        State({"type": "param", "index": ALL}, "value"),
    ],
)
def generate_graph(n_clicks, n_simulation, sample_size, distribution, values):
    sample_size = int(10 ** (sample_size))
    if (len(values) != 0) & (None not in values):
        if distribution == "uniform":
            x_tot = np.random.uniform(
                low=values[0], high=values[1], size=(n_simulation, sample_size)
            )
            mu = np.mean(values)
            sigma = np.sqrt((values[1] - values[0]) ** 2 / 12 / sample_size)
        elif distribution == "normal":
            x_tot = np.random.normal(
                loc=values[0], scale=values[1], size=(n_simulation, sample_size)
            )
            mu = values[0]
            sigma = values[1] / np.sqrt(sample_size)
        elif distribution == "beta":
            x_tot = np.random.beta(
                a=values[0], b=values[1], size=(n_simulation, sample_size)
            )
            mu = values[0] / np.sum(values)
            a = values[0]
            b = values[1]
            sigma = np.sqrt(((a * b) / ((a + b) ** 2 * (a + b + 1))) / sample_size)

        x = np.mean(x_tot, axis=1)
        max_x, min_x = np.max(x), np.min(x)
        bin_size = (
            (max_x - min_x) * 10 / len(x)
        )  # 3.49*np.std(x)*np.power(sample_size,-1/3)
        fig = ff.create_distplot(
            [x], bin_size=[bin_size], group_labels=[distribution], curve_type="kde"
        )
        print(x.shape)
        dist = norm(mu, sigma)
        x_pdf = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 100)
        fig.add_trace(
            go.Scatter(
                x=x_pdf, y=dist.pdf(x_pdf), name="CLT", line=dict(color="red", width=4)
            )
        )
        fig.update_layout(
            height=600,
            title="Mean: {0}, CI 95%: [{1}, {2}] ".format(
                np.round(mu, 2),
                np.round(mu - 2 * sigma, 2),
                np.round(mu + 2 * sigma, 2),
            ),
        )
        return fig, "Sample size (n): {0}".format(sample_size)
    else:
        return [], "Sample size (n): {0}".format(sample_size)


if __name__ == "__main__":
    server.run(debug=True, host="0.0.0.0")
