import numpy                as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs    as go
from sklearn.datasets import make_moons
from sklearn.tree     import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# fixed parameters
n_iterations = 20

def initialize(tree_depth=2, sample_size=300, sample_noise=0.2):

    # create dataset
    X, y = make_moons(n_samples=sample_size, noise=sample_noise)

    # fit classifier
    adaboost = AdaBoostClassifier(n_estimators=n_iterations, 
                                  base_estimator=DecisionTreeClassifier(max_depth=tree_depth))
    adaboost.fit(X, y)

    # get estimators in the ensemble
    estimators = adaboost.estimators_

    # get sample weights
    staged_classification = np.array(list(adaboost.staged_predict(X)))
    staged_missclassified = staged_classification != y
    staged_sample_weights = np.ones(shape=(n_iterations+1, len(X))) / len(X)
    for istage in range(1, n_iterations+1):
        estimator_weight = adaboost.estimator_weights_[istage-1]
        sample_weight = staged_sample_weights[istage-1].copy()
        incorrect = staged_missclassified[istage-1]
        ############ code snippets from sklearn AdaboostClassifier source ############
        # Only boost positive weights
        sample_weight *= np.exp(estimator_weight * incorrect *
                                        ((sample_weight > 0) |
                                        (estimator_weight < 0)))
        ##############################################################################
        sample_weight /= np.sum(sample_weight)
        staged_sample_weights[istage] = sample_weight

    # prepare to plot decision boundary
    h = .02
    xrange = np.max(X[:, 0]) - np.min(X[:, 0])
    yrange = np.max(X[:, 1]) - np.min(X[:, 1])
    xs = np.arange(np.min(X[:, 0])-xrange*0.1, np.max(X[:, 0])+xrange*0.1, h)
    ys = np.arange(np.min(X[:, 1])-yrange*0.1, np.max(X[:, 1])+yrange*0.1, h)
    xx, yy = np.meshgrid(xs, ys)
    staged_zz = np.array(list(adaboost.staged_predict(np.c_[xx.ravel(), yy.ravel()])))
    staged_zz = staged_zz.reshape(len(staged_zz), xx.shape[0], xx.shape[1])

    globalvars = {}
    globalvars['X'] = X
    globalvars['y'] = y
    globalvars['estimators'] = estimators
    globalvars['staged_sample_weights'] = staged_sample_weights
    globalvars['xs'] = xs
    globalvars['ys'] = ys
    globalvars['xx'] = xx
    globalvars['yy'] = yy
    globalvars['staged_zz'] = staged_zz
    return globalvars

globalvars = initialize()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('AdaBoost Visualized'),
    html.Div([
        html.Div([
            html.Label('Tree Depth'),
            dcc.Dropdown(
                id='tree-depth-dropdown',
                options=[
                    {'label': '1', 'value': '1'},
                    {'label': '2', 'value': '2'},
                    {'label': '3', 'value': '3'}
                ],
                value='2'
            ),
        ], style={'width': '12%', 'display': 'inline-block'}),
        html.Div([
            html.Label('Sample Size'),
            dcc.Dropdown(
                id='sample-size-dropdown',
                options=[
                    {'label': '100', 'value': '100'},
                    {'label': '300', 'value': '300'},
                    {'label': '900', 'value': '900'}
                ],
                value='300'
            ),
        ], style={'width': '12%', 'display': 'inline-block'}),
        html.Div([
            html.Label('Sample Noise'),
            dcc.Dropdown(
                id='sample-noise-dropdown',
                options=[
                    {'label': '0.1', 'value': '0.1'},
                    {'label': '0.2', 'value': '0.2'},
                    {'label': '0.3', 'value': '0.3'}
                ],
                value='0.2'
            ),
        ], style={'width': '12%', 'display': 'inline-block'})
    ]),

    html.Div([
        html.Div([], style={'width': '12%', 'display': 'inline-block'}),
        html.Div([
            html.Br(),
            html.Button(id='refit-button', n_clicks=0, children='Refit AdaBoost')
        ], style={'width': '20%', 'display': 'inline-block'})
    ]),

    html.Div([
        html.Div([
            dcc.Graph(
                id='decision-boundary',
                config={'displayModeBar': False}
                ),
        ], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(
                id='sample-weights',
                config={'displayModeBar': False}
                ),
        ], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(
                id='next-classifier',
                config={'displayModeBar': False}
                ),
        ], style={'width': '33%', 'display': 'inline-block'})
    ]),

    html.Label('Number of Iterations'),
    dcc.Slider(
        id='iteration-slider',
        min=1,
        max=n_iterations,
        value=1,
        marks={str(iter): str(iter) for iter in range(n_iterations)}
    )
])

@app.callback(dash.dependencies.Output('iteration-slider', 'value'),
              [dash.dependencies.Input('refit-button', 'n_clicks')],
              [dash.dependencies.State('tree-depth-dropdown', 'value'),
               dash.dependencies.State('sample-size-dropdown', 'value'),
               dash.dependencies.State('sample-noise-dropdown', 'value')])
def reinitialize(n_clicks, tree_depth, sample_size, sample_noise):
    global globalvars
    globalvars = initialize(tree_depth=int(tree_depth),
                            sample_size=int(sample_size),
                            sample_noise=float(sample_noise))
    return 1


@app.callback(
    dash.dependencies.Output('decision-boundary', 'figure'),
    [dash.dependencies.Input('iteration-slider', 'value')])
def update_decision_boundary_with_iter(selected_iter):

    X,y = globalvars['X'], globalvars['y']
    xs, ys, staged_zz = globalvars['xs'], globalvars['ys'], globalvars['staged_zz']

    data   = [go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                         marker=dict(color=y, colorscale='RdBu', opacity=0.7, size=5),
                         hoverinfo='none'),
              go.Heatmap(x=xs, y=ys, z=staged_zz[selected_iter-1], 
                         colorscale='RdBu', opacity=0.3, showscale=False,
                         hoverinfo='none')]
    layout = go.Layout(title='Decision Boundary', 
                       autosize=False, width=400, height=400)

    return {
        'data': data,
        'layout': layout
    }

@app.callback(
    dash.dependencies.Output('sample-weights', 'figure'),
    [dash.dependencies.Input('iteration-slider', 'value')])
def update_sample_weights_with_iter(selected_iter):

    X,y = globalvars['X'], globalvars['y']
    xs, ys, staged_zz = globalvars['xs'], globalvars['ys'], globalvars['staged_zz']
    staged_sample_weights = globalvars['staged_sample_weights']

    data   = [go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                         marker=dict(color=y, colorscale='RdBu', line=dict(width=0), opacity=0.7,
                                     size=np.sqrt(staged_sample_weights[selected_iter]*3000)),
                         hoverinfo='none'),
              go.Heatmap(x=xs, y=ys, z=staged_zz[selected_iter-1], 
                         colorscale='RdBu', opacity=0.3, showscale=False,
                         hoverinfo='none')]
    layout = go.Layout(title='Sample Weights', 
                       autosize=False, width=400, height=400)

    return {
        'data': data,
        'layout': layout
    }

@app.callback(
    dash.dependencies.Output('next-classifier', 'figure'),
    [dash.dependencies.Input('iteration-slider', 'value')])
def update_next_classifier_with_iter(selected_iter):

    X,y = globalvars['X'], globalvars['y']
    xs, ys, xx, yy = globalvars['xs'], globalvars['ys'], globalvars['xx'], globalvars['yy']
    staged_sample_weights = globalvars['staged_sample_weights']
    estimators = globalvars['estimators']

    try:
        next_estimator = estimators[selected_iter]
        next_zz = next_estimator.predict(np.c_[xx.ravel(), yy.ravel()])
        next_zz = next_zz.reshape(xx.shape)

        data = [go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                           marker=dict(color=y, colorscale='RdBu', line=dict(width=0), opacity=0.7,
                                       size=np.sqrt(staged_sample_weights[selected_iter]*3000)),
                           hoverinfo='none'),
                go.Heatmap(x=xs, y=ys, z=next_zz, 
                           colorscale='RdBu', opacity=0.3, showscale=False,
                           hoverinfo='none')]
    except:
        data = [go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                           marker=dict(color=y, colorscale='RdBu', line=dict(width=0), opacity=0.7,
                                       size=np.sqrt(staged_sample_weights[selected_iter]*3000)),
                           hoverinfo='none')]
    layout = go.Layout(title='Next Classifier', 
                       autosize=False, width=400, height=400)

    return {
        'data': data,
        'layout': layout
    }

if __name__ == '__main__':
    app.run_server(debug=True)