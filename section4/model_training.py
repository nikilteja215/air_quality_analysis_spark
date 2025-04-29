from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# 1. Start Spark Session
spark = SparkSession.builder \
    .appName("AirQualityModelTraining") \
    .getOrCreate()

# 2. Read the feature engineered data
input_path = "section2/output/feature_engineered_data/*.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

# 3. Select useful features
feature_cols = ["PM2_5_lag1", "temperature", "humidity", "PM2_5_rate_change"]

# Remove rows where PM2_5_lag1 or PM2_5_rate_change is null (first row has nulls)
df = df.na.drop(subset=["PM2_5_lag1", "PM2_5_rate_change"])

print(f"✅ Rows available for training/prediction: {df.count()}")


# 4. Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_df = assembler.transform(df)

# 5. Train/test split
train_df, test_df = assembled_df.randomSplit([0.8, 0.2], seed=42)

# 6. Linear Regression setup
lr = LinearRegression(featuresCol="features", labelCol="PM2_5")

# 7. Hyperparameter tuning
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = RegressionEvaluator(labelCol="PM2_5", predictionCol="prediction", metricName="rmse")

cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=3)

# 8. Model training
cv_model = cv.fit(train_df)

# 9. Evaluate on test set
predictions = cv_model.transform(test_df)
rmse = evaluator.evaluate(predictions)
r2 = RegressionEvaluator(labelCol="PM2_5", predictionCol="prediction", metricName="r2").evaluate(predictions)

print(f"✅ Model Training Complete!")
print(f"📈 Test RMSE: {rmse}")
print(f"📈 Test R²: {r2}")

# 10. Save predictions as CSV
predictions.select("timestamp", "region", "PM2_5", "prediction") \
    .write.mode("overwrite") \
    .option("header", True) \
    .csv("section4/output/predictions")

# 11. Save best model
cv_model.bestModel.save("section4/output/best_model")

from pyspark.sql import Row

# 12. Save evaluation metrics to a CSV file
metrics = spark.createDataFrame([
    Row(metric="RMSE", value=rmse),
    Row(metric="R2", value=r2)
])
metrics.coalesce(1).write.mode("overwrite").option("header", True).csv("section4/output/metrics/")


spark.stop()
