from pyspark.sql import SparkSession


spark = (
    SparkSession.builder
    .appName("LastFMChallenge")
    .master("local[*]")
    .getOrCreate()
)

path = "/data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"

df = (
    spark.read
    .option("sep", "\t")
    .option("header", "false")
    .csv(path)
)

print("Number of columns:", len(df.columns))

df.show(5, truncate=False)

spark.stop()