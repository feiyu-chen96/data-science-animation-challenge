import numpy                as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs    as go
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier

# set parameters
n_iterations = 20
n_samples    = 300
noise        = 0.2

# create dataset
X, y = make_moons(n_samples=n_samples, noise=noise)

# fit classifier
adaboost = AdaBoostClassifier(n_estimators=n_iterations)
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
ys = np.arange(np.min(X[:, 1])-xrange*0.1, np.max(X[:, 1])+xrange*0.1, h)
xx, yy = np.meshgrid(xs, ys)
staged_zz = np.array(list(adaboost.staged_predict(np.c_[xx.ravel(), yy.ravel()])))
staged_zz = staged_zz.reshape(len(staged_zz), xx.shape[0], xx.shape[1])




app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='decision-boundary'),
    ], style={'width': '33%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='sample-weights'),
    ], style={'width': '33%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='next-classifier'),
    ], style={'width': '33%', 'display': 'inline-block'}),
    dcc.Slider(
        id='iteration-slider',
        min=1,
        max=n_iterations,
        value=1,
        marks={str(iter): str(iter) for iter in range(n_iterations)}
    )
])

@app.callback(
    dash.dependencies.Output('decision-boundary', 'figure'),
    [dash.dependencies.Input('iteration-slider', 'value')])
def update_decision_boundary(selected_iter):

    data   = [go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                         marker=dict(color=y, colorscale='RdBu', opacity=0.7, size=5)),
              go.Heatmap(x=xs, y=ys, z=staged_zz[selected_iter-1], 
                         colorscale='RdBu', opacity=0.3, showscale=False)]
    layout = go.Layout(title='Decision Boundary', 
                       autosize=False, width=400, height=400)

    return {
        'data': data,
        'layout': layout
    }

@app.callback(
    dash.dependencies.Output('sample-weights', 'figure'),
    [dash.dependencies.Input('iteration-slider', 'value')])
def update_sample_weights(selected_iter):

    data   = [go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                         marker=dict(color=y, colorscale='RdBu', line=dict(width=0), opacity=0.7,
                                     size=np.sqrt(staged_sample_weights[selected_iter]*3000))),
              go.Heatmap(x=xs, y=ys, z=staged_zz[selected_iter-1], 
                         colorscale='RdBu', opacity=0.3, showscale=False)]
    layout = go.Layout(title='Sample Weights', 
                       autosize=False, width=400, height=400)

    return {
        'data': data,
        'layout': layout
    }

@app.callback(
    dash.dependencies.Output('next-classifier', 'figure'),
    [dash.dependencies.Input('iteration-slider', 'value')])
def update_next_classifier(selected_iter):

    try:
        next_estimator = estimators[selected_iter]
        next_zz = next_estimator.predict(np.c_[xx.ravel(), yy.ravel()])
        next_zz = next_zz.reshape(xx.shape)

        data = [go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                           marker=dict(color=y, colorscale='RdBu', line=dict(width=0), opacity=0.7,
                                       size=np.sqrt(staged_sample_weights[selected_iter]*3000))),
                go.Heatmap(x=xs, y=ys, z=next_zz, 
                            colorscale='RdBu', opacity=0.3, showscale=False)]
    except:
        data = [go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                           marker=dict(color=y, colorscale='RdBu', line=dict(width=0), opacity=0.7,
                                       size=np.sqrt(staged_sample_weights[selected_iter]*3000)))]
    layout = go.Layout(title='Next Classifier', 
                       autosize=False, width=400, height=400)

    return {
        'data': data,
        'layout': layout
    }

if __name__ == '__main__':
    app.run_server(debug=True)