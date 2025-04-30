import pandas as pd
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Create output folder if it doesn't exist
output_dir = "section5/output"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load CSV files
csv_files = glob.glob("section2/output/feature_engineered_data/part-*.csv")
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Step 2: Timestamp parsing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')

# Step 3: AQI Classification
def classify_aqi(pm):
    if pm <= 50:
        return "Good"
    elif pm <= 100:
        return "Moderate"
    else:
        return "Unhealthy"

df['AQI_Category'] = df['PM2_5'].apply(classify_aqi)

# Step 4: Line Chart - Actual vs Lag
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['PM2_5'], mode='lines', name='Actual PM2.5'))
fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['PM2_5_lag1'], mode='lines', name='Lagged PM2.5'))
fig1.update_layout(title="PM2.5: Actual vs Lagged", xaxis_title="Time", yaxis_title="PM2.5")
fig1.write_image(os.path.join(output_dir, "pm25_actual_vs_lagged.png"))

# Step 5: Spike Events
spikes = df[df['PM2_5'] > 100]
fig2 = px.scatter(spikes, x='timestamp', y='PM2_5', color='AQI_Category',
                  title="Spike Events: PM2.5 > 100")
fig2.write_image(os.path.join(output_dir, "spike_events.png"))

# Step 6: AQI Breakdown Pie
fig3 = px.pie(df, names='AQI_Category', title="AQI Category Proportions")
fig3.write_image(os.path.join(output_dir, "aqi_pie_chart.png"))

# Step 7: Correlation Matrix
corr = df[['PM2_5', 'temperature', 'humidity']].corr()
fig4 = px.imshow(corr, text_auto=True, title="Correlation Matrix")
fig4.write_image(os.path.join(output_dir, "correlation_matrix.png"))

# Step 8: Save final dashboard data
df.to_csv(os.path.join(output_dir, "dashboard_data.csv"), index=False)

print("✅ All charts and data saved as PNG images to section5/output/")
