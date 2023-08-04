"""
My futal attempt to see how nonlinear the units are over layers.

Explanation
-----------
We windowed the top bars of the units and compare the responses of the windowed
bars to the original bars (see 4a_windowed_mapping_test2.py). If the unit is
linear, then we expect the sum of the windowed responses to be the same as the
original response. In certains cases, we found that some units seem to exhibit
'Gestalt' behavior, where the original bar is greater than the sum of the
individual windowed bars. This script plots some scatter plots and histograms
to visualize the nonlinearity of the units.

Tony Fu, June 2023
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression

sys.path.append('../../..')
from src.rf_mapping.bar import *
from src.rf_mapping.model_utils import ModelInfo

# Main script
# Please specify some details here:
MODEL_NAME = 'alexnet'
TOP_RANK = 0   # If 0, then the top bar of the unit will be used, etc.
ALPHA = 4.0

# Please specify the output directory and the pdf path
output_dir = f"{c.RESULTS_DIR}/rfmp4a_windowed/{MODEL_NAME}/tests"
pdf_path = os.path.join(output_dir, f"{MODEL_NAME}_windowed_bars_test_nonlinearity.pdf")
interactdive_path = os.path.join(output_dir, f"{MODEL_NAME}_windowed_bars_test_nonlinearity.html")

# Get model information
model_info = ModelInfo()
LAYERS = model_info.get_layer_names(MODEL_NAME)

windowed_cols = list(get_coordinates_of_edges_and_corners(0,0,0,0,0,1,1).keys())

with PdfPages(pdf_path) as pdf:  
    plt.figure(figsize=(len(LAYERS)*5, 5))
    for i, layer_name in enumerate(LAYERS):
        txt_path = os.path.join(output_dir, f"{layer_name}_windowed_bars_test2.txt")
        df = pd.read_csv(txt_path, sep='\s+')

        df['windowed_response_total'] = df[windowed_cols].sum(axis=1)
        
        # Fit linear regression model
        x = df['original'].values.reshape(-1,1)
        y = df['windowed_response_total'].values.reshape(-1,1)
        model = LinearRegression().fit(x, y)
        
        # Compute predicted values for linear model line
        x_line = np.linspace(-50, 100, 400).reshape(-1, 1)
        y_line = model.predict(x_line)
        
        plt.subplot(1, len(LAYERS), i+1)
        plt.scatter(df['original'], df['windowed_response_total'])
        plt.plot(x_line, y_line, 'r')
        plt.title(f"{layer_name} (y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f})")
        plt.xlabel('original response')
        plt.ylabel('windowed response')
        plt.xlim([-50, 100])
        plt.ylim([-50, 100])
        
        # Draw crosshair at (0, 0)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.axvline(0, color='k', linewidth=0.5)

    pdf.savefig()
    plt.show()
    plt.close()

######################### Make interactive plot ###############################

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Main script and your previous code here

# Create subplots: 2 rows and a column for each layer
fig = make_subplots(rows=2, cols=len(LAYERS), subplot_titles=[f'{layer}_scatter' for layer in LAYERS] + [f'{layer}_line' for layer in LAYERS])

for i, layer_name in enumerate(LAYERS):
    txt_path = os.path.join(output_dir, f"{layer_name}_windowed_bars_test2.txt")
    df = pd.read_csv(txt_path, sep='\s+')
    df['windowed_response_total'] = df[windowed_cols].sum(axis=1)
    
    # Create scatter plot
    scatter_plot = go.Scatter(
        x=df['original'], 
        y=df['windowed_response_total'], 
        mode='markers',
        marker=dict(color='blue'),
        hovertemplate=
        '<i>Unit</i>: %{text}' +
        '<br><b>x</b>: %{x}<br>' +
        '<b>y</b>: %{y}',
        text=[f"{i}" for i in df.index]
    )
    
    # Fit linear regression model
    x = df['original'].values.reshape(-1,1)
    y = df['windowed_response_total'].values.reshape(-1,1)
    model = LinearRegression().fit(x, y)
    
    # Compute predicted values for linear model line
    x_line = np.linspace(-50, 100, 400).reshape(-1, 1)
    y_line = model.predict(x_line)
    
    # Create regression line
    regression_line = go.Scatter(
        x=x_line.flatten(), 
        y=y_line.flatten(),
        mode='lines',
        line=dict(color='red')
    )
    
    # Create crosshair at (0, 0)
    crosshair_x = go.Scatter(x=[0, 0], y=[-50, 100], mode='lines', line=dict(color='black', width=1, dash='dash'))
    crosshair_y = go.Scatter(x=[-50, 100], y=[0, 0], mode='lines', line=dict(color='black', width=1, dash='dash'))
    
    # Add the traces to the subplot
    fig.add_trace(scatter_plot, row=1, col=i+1)
    fig.add_trace(regression_line, row=1, col=i+1)
    fig.add_trace(crosshair_x, row=1, col=i+1)
    fig.add_trace(crosshair_y, row=1, col=i+1)

    # Update the subplot title with the regression formula
    fig.layout.annotations[i]['text'] = f"{layer_name}<br>(y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f})"
    
    # Create histogram
    normalized_values = df['windowed_response_total']/df['original']
    line_plot = go.Scatter(
        x=normalized_values,
        y=[0]*len(df),
        mode='markers',
        marker=dict(color='blue'),
        hovertemplate=
        '<i>Unit</i>: %{text}' +
        '<br><b>x</b>: %{x}',
        text=[f"{i}" for i in df.index]
    )
    
    fig.add_trace(line_plot, row=2, col=i+1)
    
    # Update x and y axis labels
    fig.update_xaxes(title_text="original response", row=1, col=i+1)
    fig.update_yaxes(title_text="windowed response", row=1, col=i+1)
    fig.update_xaxes(title_text="normalized windowed response", row=2, col=i+1)

fig.update_layout(height=800, width=1000*len(LAYERS), title_text="Scatter Plots and Histograms")
fig.write_html(interactdive_path)
