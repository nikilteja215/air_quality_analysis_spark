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
