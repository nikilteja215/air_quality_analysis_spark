# Air Quality Analysis using Apache Spark

## ✅ Section 1: Data Ingestion and Initial Pre-Processing

## Objective
In this section, we ingested raw air quality data from a TCP server, parsed it, structured it properly, converted datatypes, and saved it cleanly into CSV files for further processing.

## Steps Performed

### 1. TCP Server Setup
- **File:** `ingestion/tcp_log_file_streaming_server.py`
- **Purpose:** Simulate a live data stream by sending records over TCP.
- **Command Used:**
  ```bash
  python ingestion/tcp_log_file_streaming_server.py
  ```
- **Note:** Wait for "Client connected" and "Reading from" messages before proceeding.

### 2. Spark Streaming Client
- **File:** `ingestion/spark_streaming_ingestion.py`
- **Purpose:** Read data from the TCP server, clean it, and store it into CSVs.
- **Code Logic:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, to_timestamp

# 1. Start Spark Session
spark = SparkSession.builder \
    .appName("AirQualityStreamingCSV") \
    .getOrCreate()

# 2. Connect to TCP Server
tcp_host = "localhost"     # TCP server hostname
tcp_port = 9999            # TCP server port

raw_df = spark.readStream \
    .format("socket") \
    .option("host", tcp_host) \
    .option("port", tcp_port) \
    .load()

# 3. Parse and Structure Data
parsed_df = raw_df.select(
    split(raw_df.value, ",").getItem(0).alias("timestamp"),
    split(raw_df.value, ",").getItem(1).alias("region"),
    split(raw_df.value, ",").getItem(2).alias("PM2_5"),
    split(raw_df.value, ",").getItem(3).alias("temperature"),
    split(raw_df.value, ",").getItem(4).alias("humidity")
)

# 4. Convert Columns to Correct Data Types
clean_df = parsed_df.withColumn("timestamp", to_timestamp(col("timestamp"))) \
                    .withColumn("PM2_5", col("PM2_5").cast("double")) \
                    .withColumn("temperature", col("temperature").cast("double")) \
                    .withColumn("humidity", col("humidity").cast("double"))

# 5. Write to CSV files continuously
query = (
    clean_df.writeStream
    .format("csv")
    # Save cleaned data as CSV
    .option("path", "ingestion/clean_data_csv/")
    # Streaming checkpoint folder
    .option("checkpointLocation", "ingestion/checkpoint_dir_csv/")
    # Include column headers
    .option("header", "true")
    .outputMode("append")
    .start()
)

query.awaitTermination()
  ```

- **Command Used to Run:**
  ```bash
  python ingestion/spark_streaming_ingestion.py
  ```

### 3. Changes in Repository Structure
- Moved the output files (`clean_data_csv` and `checkpoint_dir_csv`) to `section1/output/`.
- Organized the repository into logical sections.

### 4. Git Commands Used
```bash
git add ingestion/tcp_log_file_streaming_server.py
git add ingestion/spark_streaming_ingestion.py
git add section1/output/clean_data_csv/*.csv
git add section1/output/checkpoint_dir_csv/*
git commit -m "✅ Section 1 complete: Nikil Teja."
git push origin master
```

## Output Folder Structure After Section 1
```
section1/
├── output/
    ├── clean_data_csv/  # Contains cleaned CSV files.
    └── checkpoint_dir_csv/  # Streaming checkpoint files.
```

## Sample Output (CSV Schema)
| timestamp                  | region  | PM2_5 | temperature | humidity |
|-----------------------------|---------|-------|-------------|----------|
| 2004-03-10T18:00:00.000Z    | Region1 | 1268  | 13          | 6        |
| 2004-03-10T19:00:00.000Z    | Region1 | 972   | 13          | 3        |

## Important Notes
- `.crc` files are generated automatically by Spark. We cleaned them manually afterward.
- Only **CSV files** were pushed to GitHub; checkpoint files are only local.

---
✨ **Section 1 Successfully Completed!** ✨

Next: Moving to **Section 2: Feature Engineering and Transformations** 🚀


# ✅ Section 2: Data Aggregations, Transformations & Trend Analysis

## Goal
Improve data quality by:
- Handling missing values and outliers
- Normalizing key features
- Adding rolling statistics, lag features, and rate-of-change

---

## 1. Input Data
The input data was **feature-engineered** from Section 1.

**Input file location:**
```bash
section1/output/clean_data_csv/*.csv
```

**Input Data Columns:**
```
timestamp, region, PM2_5, temperature, humidity
```

---

## 2. Code Used
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, when, lag
from pyspark.sql.window import Window

# Start Spark Session
spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

# Read input data
input_path = "section1/output/clean_data_csv/*.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

# Convert columns to correct types
df = df.withColumn("PM2_5", col("PM2_5").cast("double")) \
       .withColumn("temperature", col("temperature").cast("double")) \
       .withColumn("humidity", col("humidity").cast("double"))

# Handle missing values (Mean Imputation)
for feature in ["PM2_5", "temperature", "humidity"]:
    mean_val = df.select(avg(col(feature))).first()[0]
    df = df.fillna({feature: mean_val})

# Handle outliers (Cap at mean ± 3*stddev)
for feature in ["PM2_5", "temperature", "humidity"]:
    stats = df.select(avg(col(feature)), stddev(col(feature))).first()
    mean_val, std_val = stats
    lower_bound = mean_val - (3 * std_val)
    upper_bound = mean_val + (3 * std_val)
    df = df.withColumn(feature,
        when((col(feature) < lower_bound) | (col(feature) > upper_bound), mean_val).otherwise(col(feature))
    )

# Normalize features (Z-Score Normalization)
for feature in ["PM2_5", "temperature", "humidity"]:
    stats = df.select(avg(col(feature)), stddev(col(feature))).first()
    mean_val, std_val = stats
    df = df.withColumn(f"{feature}_zscore", (col(feature) - mean_val) / std_val)

# Create lag feature and rate-of-change
windowSpec = Window.orderBy("timestamp")
df = df.withColumn("PM2_5_lag1", lag("PM2_5", 1).over(windowSpec))
df = df.withColumn("PM2_5_rate_change", 
        ((col("PM2_5") - col("PM2_5_lag1")) / col("PM2_5_lag1"))
    )

# Fill NaN created by lag with 0.0
df = df.fillna({"PM2_5_lag1": 0.0, "PM2_5_rate_change": 0.0})

# Save output
output_path = "section2/output/feature_engineered_data/"
df.write.mode("overwrite").option("header", True).csv(output_path)

print("✅ Feature engineering complete!")

spark.stop()
```

---

## 3. Output Data

**Output Directory:**
```bash
section2/output/feature_engineered_data/
```

**Output Columns:**
```
timestamp, region, PM2_5, temperature, humidity,
PM2_5_zscore, temperature_zscore, humidity_zscore,
PM2_5_lag1, PM2_5_rate_change
```

---

## 4. Key Improvements Done

- ✅ Imputed missing values with mean
- ✅ Capped outliers to mean ± 3×std deviation
- ✅ Z-score normalization of PM2_5, temperature, and humidity
- ✅ Added `PM2_5_lag1` (previous timestamp value)
- ✅ Added `PM2_5_rate_change` (percentage change from previous)

---

## 5. Commands Used

**To run feature engineering code:**
```bash
python section2/feature_engineering.py
```

**To push Section 2 to GitHub:**
```bash
git add section2/
git commit -m "✅ Section 2 complete: Bhargavi Potu."
git push origin master
```

---

✨ **Section 2 Successfully Completed!** ✨

Next: Moving to **Section 3: Spark SQL Exploration & Correlation Analysis** 🚀

**Section 3** 

# Air Quality Analysis with Apache Spark

This task performs data-driven air quality analysis using Apache Spark, focusing on PM2.5 measurements. It includes data cleaning, aggregation, correlation analysis, and classification of air quality risk based on AQI standards.


## 🚀 Section 3: Spark SQL Exploration & Correlation Analysis

This module focuses on deeper insights from air quality data using PySpark SQL.

### ✅ Features Implemented:
- Register cleaned data as a Spark SQL view
- Run complex SQL queries to identify pollution trends and hotspots
- Use window functions (`LAG`, `LEAD`) for trend analysis
- Classify regions using a simplified AQI UDF
- Output all results to `section3/output/`

---

### 🔧 How to Run

#### 🖥️ Prerequisites
- Apache Spark installed
- Java installed and `JAVA_HOME` set
- PySpark accessible via terminal

#### 📦 Install Requirements (if needed)

pip install pyspark
#### Run the script
spark-submit section3/section3_air_quality_analysis.py

File:Folder	Description
highest_avg_pm25_24h: Regions with the highest PM2.5 over the last 24 hours
peak_pollution_intervals:	Hours of day with highest PM2.5 per region
pm25_trend_increase:	Sudden or sustained PM2.5 spikes detected using window functions
aqi_classification_summary:	Region-based AQI classification: Good, Moderate, Unhealthy
**Section 3 Successfully Completed!** 

# 📊 Section 5: Dashboard Visualization - Air Quality Monitoring & Forecasting

This module visualizes feature-engineered air quality data to detect pollution trends, classify AQI categories, and analyze environmental correlations. Outputs are saved as high-quality PNG images for static reporting and analysis.

---

## ✅ Features Implemented

| Step | Description |
|------|-------------|
| 1️⃣ | Load and combine feature-engineered CSV files from Section 2 |
| 2️⃣ | Parse timestamps and sort data chronologically |
| 3️⃣ | Classify PM2.5 values into AQI categories (`Good`, `Moderate`, `Unhealthy`) |
| 4️⃣ | Plot actual vs lagged PM2.5 over time (line chart) |
| 5️⃣ | Visualize PM2.5 spikes > 100 using a scatter plot |
| 6️⃣ | Show AQI category proportions with a pie chart |
| 7️⃣ | Compute correlation matrix (PM2.5, temperature, humidity) and display as heatmap |
| 8️⃣ | Save enriched dashboard dataset as CSV |

---

## 📂 Input

**Path:**  
`section2/output/feature_engineered_data/part-*.csv`
## output columns
timestamp, PM2_5, PM2_5_lag1, temperature, humidity, AQI_Category

All output files are saved to:  
`section5/output/`

### 📷 Visualizations (PNG):
- `pm25_actual_vs_lagged.png`
- `spike_events.png`
- `aqi_pie_chart.png`
- `correlation_matrix.png`

### 📄 CSV:
- `dashboard_data.csv`
## 🧪 How to Run
python section5/pipeline_dashboard.py





