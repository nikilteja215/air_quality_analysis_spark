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