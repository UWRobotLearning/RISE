import numpy as np
import plotly.graph_objs as go
import dash
import h5py
import click
from dash import dcc, html, Output, Input


@click.command()
@click.option('--dataset', '-d', required=True, type=click.Path(exists=True), help='Path to the HDF5 dataset.')
@click.option('--sample_percentage', '-p', default=100.0, help='Percentage of states to sample (0-100).')
def main(dataset, sample_percentage):

    # ---------------------------
    # 1. Generate (or load) data
    # ---------------------------
    with h5py.File(dataset, 'r') as f:
        data = f['data']

        # gather states and actions from all episodes
        states_list = []
        actions_list = []

        for ep_key in data:
            ep = data[ep_key]
            # Replace this with your actual dataset fields for (X, Y, Z) state data
            ep_states = ep['obs']['robot0_eef_pos'][:]
            # valid_indices = np.where((ep_states[:, 0] < -0.1) & (ep_states[:, 1] > 0.1))[0]
            filtered_states = ep_states[:]
            filtered_actions = ep['actions'][:, :3][:]

            states_list.append(filtered_states)
            actions_list.append(filtered_actions)

        # combine all episodes into single arrays
        states = np.concatenate(states_list, axis=0)
        actions = np.concatenate(actions_list, axis=0)

    # # Sample only p% of the states if needed
    if sample_percentage < 100.0:
        p = sample_percentage / 100.0
        total_states = len(states)
        sample_size = int(total_states * p)
        indices = np.random.choice(total_states, sample_size, replace=False)
        states = states[indices]
        actions = actions[indices]

    # ---------------------------------------
    # 2. Function to create an ellipsoid mesh
    # ---------------------------------------
    def create_ellipsoid_mesh(center, radii, u_steps=12, v_steps=12):
        """
        Create meshgrid coordinates for an ellipsoid.
        center: 3-element array-like (center of ellipsoid)
        radii:  3-element array-like (radii along x,y,z)
        Returns: x, y, z mesh arrays.
        """
        u = np.linspace(0, 2 * np.pi, u_steps)
        v = np.linspace(0, np.pi, v_steps)
        u, v = np.meshgrid(u, v)
        x = radii[0] * np.cos(u) * np.sin(v) + center[0]
        y = radii[1] * np.sin(u) * np.sin(v) + center[1]
        z = radii[2] * np.cos(v) + center[2]
        return x, y, z

    # ---------------------------
    # 3. Build the Dash App Layout
    # ---------------------------
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("State–Action Distribution Visualization"),
        html.Div([
            dcc.Graph(
                id='state-scatter', 
                config={'displayModeBar': False},
                style={'height': '700px'}  # Increased height
            ),
            html.P("Click on a state to select it for querying.")
        ]),
        html.Div([
            html.Label("Neighborhood Threshold (in state space)"),
            dcc.Slider(
                id='threshold-slider',
                min=0.01,
                max=1.0,
                step=0.01,
                value=0.05,
                marks={i: f"{i:.1f}" for i in np.arange(0.1, 5.1, 0.5)}
            )
        ], style={'width': '80%', 'margin': '20px auto'}),
        html.Div([
            dcc.Graph(
                id='action-ellipsoids', 
                config={'displayModeBar': False},
                style={'height': '700px'}  # Increased height
            ),
            html.P("Marginal action distribution p(dx,dy,dz | s) as a blob of ellipsoids.")
        ])
    ], style={'overflowY': 'auto', 'height': '1500px'})  # Allows scrolling

    # -----------------------------------------
    # 4. Callback: Update the 3D state scatter plot
    # -----------------------------------------
    @app.callback(
        Output('state-scatter', 'figure'),
        [Input('state-scatter', 'clickData'),
        Input('threshold-slider', 'value')]
    )
    def update_state_scatter(clickData, threshold):
        # Base scatter plot: all states in blue.
        trace_all = go.Scatter3d(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            mode='markers',
            marker=dict(size=4, color='blue'),
            name='All States'
        )

        # Create the initial figure.
        fig = go.Figure(data=[trace_all], layout=go.Layout(
            title="3D State Scatter (Click a state to query)",
            scene=dict(
                xaxis_title='X', 
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='auto'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        ))

        # If a state was clicked, compute and highlight the neighborhood.
        if clickData is not None:
            idx = clickData['points'][0]['pointNumber']
            selected_state = states[idx]

            # Compute Euclidean distances from the selected state.
            distances = np.linalg.norm(states - selected_state, axis=1)
            neighborhood_mask = distances <= threshold
            neighborhood_states = states[neighborhood_mask]

            # Add neighborhood states in green.
            trace_neighborhood = go.Scatter3d(
                x=neighborhood_states[:, 0],
                y=neighborhood_states[:, 1],
                z=neighborhood_states[:, 2],
                mode='markers',
                marker=dict(size=6, color='green'),
                name='Neighborhood'
            )
            fig.add_trace(trace_neighborhood)

            # Highlight the selected state in red.
            trace_selected = go.Scatter3d(
                x=[selected_state[0]],
                y=[selected_state[1]],
                z=[selected_state[2]],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Selected State'
            )
            fig.add_trace(trace_selected)
        
        return fig

    # -----------------------------------------------------------
    # 5. Callback: Update the action distribution ellipsoids plot
    # -----------------------------------------------------------
    @app.callback(
        Output('action-ellipsoids', 'figure'),
        [Input('state-scatter', 'clickData'),
        Input('threshold-slider', 'value')]
    )
    def update_action_ellipsoids(clickData, threshold):
        # If no state is selected, display an empty plot with instructions.
        if clickData is None:
            layout = go.Layout(
                title="Marginal Action Distribution p(a|s): Select a state from the above plot",
                scene=dict(xaxis_title='dx', yaxis_title='dy', zaxis_title='dz'),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            return go.Figure(layout=layout)
        
        # Determine the query state (selected state) and its neighborhood.
        idx = clickData['points'][0]['pointNumber']
        query_state = states[idx]
        dists = np.linalg.norm(states - query_state, axis=1)
        # Use a hard threshold: keep only states with distance <= threshold.
        neighborhood_mask = dists <= threshold
        neighborhood_actions = actions[neighborhood_mask]
        
        # For clarity, if many actions fall in the neighborhood, sample a subset.
        max_ellipsoids = 150
        if neighborhood_actions.shape[0] > max_ellipsoids:
            sel_idx = np.random.choice(neighborhood_actions.shape[0], max_ellipsoids, replace=False)
            sampled_actions = neighborhood_actions[sel_idx]
        else:
            sampled_actions = neighborhood_actions
        
        # Create a list of ellipsoid mesh traces.
        # Here, we use a fixed radii for each ellipsoid (you can adjust as needed).
        fixed_radii = [0.03, 0.03, 0.03]
        ellipsoid_traces = []
        for act in sampled_actions:
            x_e, y_e, z_e = create_ellipsoid_mesh(act, fixed_radii, u_steps=12, v_steps=12)
            ellipsoid = go.Surface(
                x=x_e,
                y=y_e,
                z=z_e,
                opacity=0.5,
                colorscale='Viridis',
                showscale=False,
                hoverinfo='skip'
            )
            ellipsoid_traces.append(ellipsoid)
        
        layout = go.Layout(
            title="Marginal Action Distribution p(dx,dy,dz | s)",
            scene=dict(
                xaxis_title='dx', 
                yaxis_title='dy', 
                zaxis_title='dz',
                aspectmode='auto'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        fig = go.Figure(data=ellipsoid_traces, layout=layout)
        return fig
    app.run_server(debug=True)


# ---------------------------
# 6. Run the Dash App
# ---------------------------
if __name__ == '__main__':
    main()