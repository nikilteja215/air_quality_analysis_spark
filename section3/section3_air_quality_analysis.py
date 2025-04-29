from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, lag
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import os

# Step 1: Initialize Spark
spark = SparkSession.builder.appName("Section3AirQuality").getOrCreate()

# Step 2: Load Input CSV File
df = spark.read.csv("section3/input/cleaned.csv", header=True, inferSchema=True)
df_cleaned = df.dropna(subset=["PM2_5", "timestamp", "region"])

# Step 3: Register Temp View
df_cleaned.createOrReplaceTempView("air_quality_data")

# Step 4: Create Output Directory
os.makedirs("section3/output", exist_ok=True)

# Step 5: Highest Avg PM2.5 in Last 24 Hours
query1 = spark.sql("""
    SELECT region, AVG(PM2_5) AS avg_pm25
    FROM air_quality_data
    WHERE timestamp >= current_timestamp() - INTERVAL 24 HOURS
    GROUP BY region
    ORDER BY avg_pm25 DESC
    LIMIT 10
""")
query1.write.mode("overwrite").csv("section3/output/highest_avg_pm25_24h", header=True)

# Step 6: Peak Pollution Intervals by Hour
df_peak = df_cleaned.withColumn("hour", hour("timestamp")) \
    .groupBy("region", "hour").avg("PM2_5") \
    .withColumnRenamed("avg(PM2_5)", "avg_pm25") \
    .orderBy("avg_pm25", ascending=False)
df_peak.write.mode("overwrite").csv("section3/output/peak_pollution_intervals", header=True)

# Step 7: PM2.5 Trend Analysis
windowSpec = Window.partitionBy("region").orderBy("timestamp")
df_trend = df_cleaned.withColumn("prev_pm25", lag("PM2_5").over(windowSpec)) \
    .withColumn("delta_pm25", col("PM2_5") - col("prev_pm25")) \
    .filter(col("delta_pm25") > 10)
df_trend.write.mode("overwrite").csv("section3/output/pm25_trend_increase", header=True)

# Step 8: AQI Classification
def classify_aqi(pm25):
    if pm25 <= 12:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    else:
        return "Unhealthy"

aqi_udf = udf(classify_aqi, StringType())
df_classified = df_cleaned.withColumn("aqi_category", aqi_udf(col("PM2_5")))
df_classified.createOrReplaceTempView("classified_air_quality")

query2 = spark.sql("""
    SELECT region, aqi_category, COUNT(*) AS count
    FROM classified_air_quality
    GROUP BY region, aqi_category
    ORDER BY aqi_category DESC, count DESC
""")
query2.write.mode("overwrite").csv("section3/output/aqi_classification_summary", header=True)

print("✅ Section 3 analysis completed. Output saved in section3/output/")
