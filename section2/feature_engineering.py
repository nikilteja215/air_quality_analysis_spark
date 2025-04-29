from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, when, lag
from pyspark.sql.window import Window

# 1. Start Spark Session
spark = SparkSession.builder \
    .appName("FeatureEngineering") \
    .getOrCreate()

# 2. Read the input data
input_path = "section1/output/clean_data_csv/*.csv"

# Read the CSV files with schema inference
df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

# 3. Convert Columns to Correct Data Types
df = df.withColumn("PM2_5", col("PM2_5").cast("double")) \
       .withColumn("temperature", col("temperature").cast("double")) \
       .withColumn("humidity", col("humidity").cast("double"))

# 4. Handle Missing Values (simple mean imputation)
for col_name in ["PM2_5", "temperature", "humidity"]:
    mean_val = df.select(avg(col(col_name))).first()[0]
    df = df.fillna({col_name: mean_val})

# 5. Handle Outliers (Cap values to mean ± 3*stddev)
for col_name in ["PM2_5", "temperature", "humidity"]:
    stats = df.select(avg(col(col_name)), stddev(col(col_name))).first()
    mean_val, std_val = stats[0], stats[1]
    lower_bound = mean_val - (3 * std_val)
    upper_bound = mean_val + (3 * std_val)
    df = df.withColumn(col_name,
        when((col(col_name) < lower_bound) | (col(col_name) > upper_bound), mean_val)
        .otherwise(col(col_name))
    )

# 6. Normalize Features (Z-score Normalization)
for col_name in ["PM2_5", "temperature", "humidity"]:
    stats = df.select(avg(col(col_name)), stddev(col(col_name))).first()
    mean_val, std_val = stats[0], stats[1]
    df = df.withColumn(f"{col_name}_zscore", (col(col_name) - mean_val) / std_val)

# 7. Add Lag and Rate-of-Change Features
window_spec = Window.partitionBy("region").orderBy("timestamp")
df = df.withColumn("PM2_5_lag1", lag("PM2_5", 1).over(window_spec))
df = df.withColumn("PM2_5_rate_change", (col("PM2_5") - col("PM2_5_lag1")) / col("PM2_5_lag1"))
# Fill null values in lag and rate change with 0.0
df = df.fillna({"PM2_5_lag1": 0.0, "PM2_5_rate_change": 0.0})

# 8. Output directory for feature engineered data
output_path = "section2/output/feature_engineered_data/"
df.write.mode("overwrite").option("header", True).csv(output_path)

print("✅ Feature engineering complete!")

# 9. Stop Spark Session
spark.stop()
