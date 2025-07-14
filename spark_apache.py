from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Step 1: Initialize Spark
spark = SparkSession.builder \
    .appName("Sign Language Keypoint Preprocessing") \
    .getOrCreate()

# Step 2: Load Dataset
df_spark = spark.read.csv("keypoints.csv", header=True, inferSchema=True)

# Step 3: Remove rows with all-zero keypoints
non_zero_df = df_spark.filter(~(sum([col(c) for c in df_spark.columns[:-1]]) == 0))

# Step 4: Normalize keypoints (0-1)
normalize_udf = udf(lambda x: float(x)/100 if x is not None else 0, DoubleType())
for colname in df_spark.columns[:-1]:
    non_zero_df = non_zero_df.withColumn(colname, normalize_udf(col(colname)))

# Step 5: Convert to Pandas for label encoding
df_cleaned = non_zero_df.toPandas()

# Step 6: Encode labels
le = LabelEncoder()
df_cleaned['label'] = le.fit_transform(df_cleaned['label'])

# Step 7: Save cleaned dataset
df_cleaned.to_csv("keypoints_spark_cleaned.csv", index=False)
print("✅ Preprocessed dataset saved using Apache Spark!")
