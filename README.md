# Air Quality Analysis using Apache Spark

## ✅ Section 1: Data Ingestion and Initial Pre-Processing

### Goals:
- Ingest raw air quality data using Spark Structured Streaming
- Parse and structure the data (PM2.5, temperature, humidity)
- Clean and merge the data into a single DataFrame

### Data Source:
Historical data was streamed over a TCP connection using a custom TCP socket server simulating live data.

### 🛠️TCP Server Commands Used:
To run the TCP server: `ingestion/tcp_log_file_streaming_server.py`
```bash
python ingestion/tcp_log_file_streaming_server.py
```
To run Spark Structured Streaming: `ingestion/spark_streaming_ingestion.py`
```bash
python ingestion/spark_streaming_ingestion.py
```

### Spark Streaming Script Used:
`ingestion/spark_streaming_ingestion.py`
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

### Final Output:
All `.csv` files were written to:
- `section1/output/clean_data_csv/`
- Checkpointing to `section1/output/checkpoint_dir_csv/`

### Git Commands:
```bash
git add section1/output/clean_data_csv/*.csv
git add section1/output/checkpoint_dir_csv
git commit -m "✅ Section 1 complete: Nikil Teja."
git push origin master
```

---

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
