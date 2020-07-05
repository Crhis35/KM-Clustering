import dash
import pandas as pd
import numpy as np
import plotly.offline as plt
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.figure_factory as ff

pd.set_option('display.max_columns', 10)

app = dash.Dash()
app.layout = html.Div([
    html.Title(children='KMC'),
    html.Div([
        html.H1(
            children="K Mean Clustering",
            style={
                'text-align': 'center',
                'color': '#7FDBFF'
            }
        ),
        html.P(children='Seleccione el numero de cluster'),
        dcc.Input(
            id="opcion",
            type="number",
            placeholder="Ingrese",
            value=7,
            min=1
        ),
        dcc.Graph(id='feature-graphic')
    ])
])


@app.callback(
    Output('feature-graphic', 'figure'),
    [Input('opcion', 'value')])
def update_graph(k):
    data = pd.read_csv('./data/dataset.csv')
    data = data[['V1', 'V2']]
    k_means = (data.sample(k, replace=False))
    k_means2 = pd.DataFrame()
    clusters = pd.DataFrame()

    while not k_means2.equals(k_means):

        # distance matrix
        cluster_count = 0
        for idx, k_mean in k_means.iterrows():
            clusters[cluster_count] = (
                data[k_means.columns] - np.array(k_mean)).pow(2).sum(1).pow(0.5)
            cluster_count += 1

        # update cluster
        data['MDCluster'] = clusters.idxmin(axis=1)

        # store previous cluster
        k_means2 = k_means
        k_means = pd.DataFrame()
        k_means_frame = data.groupby('MDCluster').agg(np.mean)

        k_means[k_means_frame.columns] = k_means_frame[k_means_frame.columns]

    traces = [go.Scatter(
        x=data['V1'],
        y=data['V2'].where(data['MDCluster'] == c),
        mode='markers',
        name='Cluster: ' + str(c)
    ) for c in range(k)]

    traces.append(
        go.Scatter(
            x=k_means['V1'],
            y=k_means['V2'],
            mode='markers',
            marker=dict(
                size=10,
                color='#000000',
            ),
            name='Centroids of Clusters'
        )
    )

    fig = {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'X'},
            yaxis={'title': 'Y'},
            hovermode='closest'
        )
    }
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
