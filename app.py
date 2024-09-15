import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from joblib import load
import logging

from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the GPT4All model
model = GPT4All(model="/models/qwen2-1_5b-instruct-q4_0.gguf", n_threads=8, max_tokens=32000)
#model = GPT4All(model="/models/qwen2-1_5b-instruct-q4_0.gguf", n_threads=8, max_tokens=16000)

# Define the prompt template
template = """
You are an AI assistant specialized in cardiology.

Given the patient data:
{patient_data}

Provide a medical summary, assess the risk of heart disease, and suggest recommendations.

Medical Summary:
"""

prompt = PromptTemplate(
    input_variables=["patient_data"],
    template=template,
)

# Compose the chain using the prompt and the model
chain = prompt | model

# Load the preprocessing objects
label_encoders = load('label_encoders.joblib')
scaler = load('scaler.joblib')

# Load the dataset for reference
url = 'heart.csv'
df = pd.read_csv(url)

# Prepare the data
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['HeartDisease']]

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout components
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Patient Data Input"),
            html.Div([
                # Numerical inputs
                *[
                    html.Div([
                        dbc.Label(col),
                        dbc.Input(id=col, type="number", placeholder=f"Enter {col}")
                    ], className="mb-3") for col in numerical_cols
                ],
                # Categorical inputs
                *[
                    html.Div([
                        dbc.Label(col),
                        dbc.Select(
                            id=col,
                            options=[{'label': val, 'value': val} for val in df[col].unique()],
                            placeholder=f"Select {col}"
                        )
                    ], className="mb-3") for col in categorical_cols
                ],
            ]),
            dbc.Button("Get Medical Summary", id="submit-button", color="primary", className="mt-3"),
        ], width=4),
        dbc.Col([
            html.H2("Medical Summary"),
            # Add a loading spinner here using dcc.Loading
            dcc.Loading(
                id="loading",
                type="graph",  # Spinner type (options: 'circle', 'cube', 'dot', 'graph')
                fullscreen=True,  # Set to True if you want the spinner to cover the entire screen
                children=[
                    html.Div([
                        # The output summary area
                        html.Div(id='output-summary', children="Medical summary will appear here.", style={'whiteSpace': 'pre-line', 'textAlign': 'center'})
                    ])
                ]
            )
        ], width=8)
    ]),
], fluid=True)

# Define callback
@app.callback(
    Output('output-summary', 'children'),
    Input('submit-button', 'n_clicks'),
    [State(col, 'value') for col in numerical_cols + categorical_cols]
)
def update_output(n_clicks, *values):
    if n_clicks is None:
        return "Medical summary will appear here."

    # Create patient data dictionary
    patient_data = dict(zip(numerical_cols + categorical_cols, values))

    # Handle None values
    if None in patient_data.values():
        return "Please fill in all the fields."

    # Create separate dictionaries
    patient_data_for_llm = patient_data.copy()   # Original values for LLM
    patient_data_for_ml = {}                     # Scaled values for ML models

    # Process numerical inputs
    numerical_values = []
    for col in numerical_cols:
        try:
            value = float(patient_data[col])
            numerical_values.append(value)
            patient_data_for_ml[col] = value      # Store original for ML if needed
        except ValueError:
            return f"Invalid input for {col}. Please enter a numerical value."

    # Create a DataFrame with the same feature names as used during fitting
    numerical_df = pd.DataFrame([numerical_values], columns=numerical_cols)

    # Scale numerical features
    scaled_array = scaler.transform(numerical_df)

    # Update patient_data_for_ml with scaled values
    for i, col in enumerate(numerical_cols):
        patient_data_for_ml[col] = scaled_array[0, i]

    # (Optional) Use scaled_array for ML models here
    # Example: prediction = ml_model.predict(scaled_array)

    # Convert patient_data_for_llm to string for LLM input using original values
    patient_data_str = '\n'.join(f"{k}: {v}" for k, v in patient_data_for_llm.items())

    # Log the patient_data_str for debugging
    logging.info(f"Patient Data for LLM:\n{patient_data_str}")

    # Generate medical summary
    try:
        response = chain.invoke({"patient_data": patient_data_str})
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return f"An error occurred while generating the medical summary: {str(e)}"

    # Log the response for debugging
    logging.info(f"Medical Summary Response:\n{response}")

    return response

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
