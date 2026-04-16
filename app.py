"""
app.py - Melbourne Housing Price Predictor
Prices displayed in ZAR (converted from AUD).
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

lr_model        = joblib.load(os.path.join(MODELS_DIR, 'lr_model.pkl'))
rf_model        = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
xgb_model       = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = 'Melbourne Property Price Predictor'
server = app.server

AUD_TO_ZAR = 12.50  # 1 AUD = R12.50

def encode_property(data_dict):
    df = pd.DataFrame([data_dict])
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_final = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_final

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1('🏡 Property Price Predictor', className='text-center my-4 text-primary'))]),
    dbc.Row([dbc.Col(html.P('Enter property details to get an estimated market price (Melbourne data, prices in ZAR).', className='text-center text-muted mb-4'))]),

    dbc.Card([
        dbc.CardHeader(html.H4('Property Details', className='mb-0')),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label('Rooms'),
                    dbc.Input(id='rooms', type='number', value=3, min=1, max=10, className='mb-3'),
                    dbc.Label('Bedrooms'),
                    dbc.Input(id='bedrooms', type='number', value=3, min=1, max=10, className='mb-3'),
                    dbc.Label('Bathrooms'),
                    dbc.Input(id='bathrooms', type='number', value=2, min=1, max=10, className='mb-3'),
                ], md=4),
                dbc.Col([
                    dbc.Label('Car Spots'),
                    dbc.Input(id='car', type='number', value=1, min=0, max=10, className='mb-3'),
                    dbc.Label('Land Size (m²)'),
                    dbc.Input(id='landsize', type='number', value=500, min=0, className='mb-3'),
                    dbc.Label('Year Built'),
                    dbc.Input(id='yearbuilt', type='number', value=1990, min=1800, max=2024, className='mb-3'),
                ], md=4),
                dbc.Col([
                    dbc.Label('Property Type'),
                    dcc.Dropdown(id='prop_type', options=[
                        {'label': 'House/Villa/Cottage', 'value': 'h'},
                        {'label': 'Unit/Duplex',         'value': 'u'},
                        {'label': 'Townhouse',           'value': 't'},
                    ], value='h', clearable=False, className='mb-3'),
                    dbc.Label('Distance from CBD (km)'),
                    dbc.Input(id='distance', type='number', value=10, min=0, className='mb-3'),
                    dbc.Label('Region'),
                    dcc.Dropdown(id='region', options=[
                        {'label': 'Northern Metropolitan',       'value': 'Northern Metropolitan'},
                        {'label': 'Southern Metropolitan',       'value': 'Southern Metropolitan'},
                        {'label': 'Eastern Metropolitan',        'value': 'Eastern Metropolitan'},
                        {'label': 'Western Metropolitan',        'value': 'Western Metropolitan'},
                        {'label': 'South-Eastern Metropolitan',  'value': 'South-Eastern Metropolitan'},
                        {'label': 'Northern Victoria',           'value': 'Northern Victoria'},
                        {'label': 'Eastern Victoria',            'value': 'Eastern Victoria'},
                        {'label': 'Western Victoria',            'value': 'Western Victoria'},
                    ], value='Southern Metropolitan', clearable=False, className='mb-3'),
                    dbc.Label('Model'),
                    dcc.Dropdown(id='model_choice', options=[
                        {'label': 'Random Forest (Recommended)', 'value': 'rf'},
                        {'label': 'XGBoost',                     'value': 'xgb'},
                        {'label': 'Linear Regression',           'value': 'lr'},
                    ], value='rf', clearable=False, className='mb-3'),
                ], md=4),
            ]),
            dbc.Row([dbc.Col([dbc.Button('💰 Predict Price', id='predict-btn', color='primary', size='lg', className='w-100 mt-2')])])
        ])
    ], className='mb-4 shadow'),

    html.Div(id='results-section'),
], fluid=True)


@app.callback(
    Output('results-section', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('rooms', 'value'), State('bedrooms', 'value'), State('bathrooms', 'value'),
    State('car', 'value'), State('landsize', 'value'), State('yearbuilt', 'value'),
    State('prop_type', 'value'), State('distance', 'value'), State('region', 'value'),
    State('model_choice', 'value'),
    prevent_initial_call=True
)
def predict(n_clicks, rooms, bedrooms, bathrooms, car, landsize, yearbuilt,
            prop_type, distance, region, model_choice):

    property_data = {
        'Rooms': rooms, 'Type': prop_type, 'Distance': distance,
        'Bedroom2': bedrooms, 'Bathroom': bathrooms, 'Car': car,
        'Landsize': landsize, 'YearBuilt': yearbuilt, 'Regionname': region,
    }
    X_input = encode_property(property_data)

    if model_choice == 'lr':
        model, model_name = lr_model, 'Linear Regression'
    elif model_choice == 'xgb':
        model, model_name = xgb_model, 'XGBoost'
    else:
        model, model_name = rf_model, 'Random Forest'

    predicted_aud = float(model.predict(X_input)[0])
    predicted_zar = predicted_aud * AUD_TO_ZAR
    low_zar  = predicted_zar * 0.90
    high_zar = predicted_zar * 1.10

    if model_choice in ('rf', 'xgb'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:8]
        top_features = [feature_columns[i] for i in indices]
        top_vals = [importances[i] for i in indices]
        imp_fig = go.Figure(go.Bar(x=top_vals[::-1], y=top_features[::-1], orientation='h', marker_color='#3498db'))
        imp_fig.update_layout(title='Top Feature Importances', xaxis_title='Importance', template='plotly_white')
        importance_section = dbc.Col([dcc.Graph(figure=imp_fig)], md=6)
    else:
        importance_section = dbc.Col([dbc.Alert('Feature importance available for Random Forest and XGBoost.', color='info')], md=6)

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number", value=predicted_zar,
        number={'prefix': 'R', 'valueformat': ',.0f'},
        title={'text': f"Predicted Price ({model_name})"},
        gauge={
            'axis': {'range': [0, 15000000]},
            'bar': {'color': '#2ecc71'},
            'steps': [
                {'range': [0, 5000000],    'color': '#eafaf1'},
                {'range': [5000000, 10000000], 'color': '#d5f5e3'},
                {'range': [10000000, 15000000], 'color': '#abebc6'},
            ],
        }
    ))
    gauge_fig.update_layout(template='plotly_white')

    prop_type_label = {'h': 'House/Villa/Cottage', 'u': 'Unit/Duplex', 't': 'Townhouse'}.get(prop_type, prop_type)

    return dbc.Card([
        dbc.CardHeader(html.H4('💰 Price Prediction Results', className='mb-0')),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([dbc.Card([dbc.CardBody([
                    html.H5('Estimated Price', className='text-muted'),
                    html.H2(f'R{predicted_zar:,.0f}', style={'color': '#2ecc71', 'fontWeight': 'bold'}),
                    html.P(f'Range: R{low_zar:,.0f} — R{high_zar:,.0f}', className='text-muted'),
                    html.P(f'Model: {model_name}', className='text-muted small'),
                ])], className='text-center shadow-sm')], md=6),
                dbc.Col([dbc.Card([dbc.CardBody([
                    html.H5('Property Summary', className='text-muted'),
                    html.P(f'🏠 Type: {prop_type_label}'),
                    html.P(f'📍 Region: {region}'),
                    html.P(f'📐 Land: {landsize:,} m² | Distance: {distance} km from CBD'),
                    html.P(f'🛏 Rooms: {rooms} | Bedrooms: {bedrooms} | 🛁 Bathrooms: {bathrooms}'),
                    html.P(f'🚗 Car spots: {car} | 🏗 Year Built: {yearbuilt}'),
                ])], className='shadow-sm')], md=6),
            ], className='mb-4'),
            dbc.Row([dbc.Col([dcc.Graph(figure=gauge_fig)], md=6), importance_section]),
        ])
    ], className='shadow')


if __name__ == '__main__':
    app.run(debug=True)
