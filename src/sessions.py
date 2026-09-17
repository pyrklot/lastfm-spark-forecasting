from pyspark.sql import functions as F
from pyspark.sql.window import Window


INPUT_PATH = (
    "/data/lastfm-dataset-1K/"
    "userid-timestamp-artid-artname-traid-traname.tsv"
)

SESSION_GAP_SECONDS = 20 * 60


def load_data(spark):

    print("Loading Last.fm data...")

    df = (
        spark.read
        .option("sep", "\t")
        .option("header", "false")
        .option("inferSchema", "false")
        .csv(INPUT_PATH)
        .toDF(
            "user_id",
            "timestamp",
            "artist_id",
            "artist_name",
            "track_id",
            "track_name",
        )
    )

    # Parse timestamps
    df = df.withColumn(
        "event_time",
        F.to_timestamp(
            "timestamp",
            "yyyy-MM-dd'T'HH:mm:ssX",
        ),
    )

    # Check timestamp quality w one aggregation
    quality = df.select(
        F.count("*").alias("total"),
        F.sum(
            F.when(
                F.col("event_time").isNotNull(),
                1,
            ).otherwise(0)
        ).alias("valid"),
    ).first()

    total = quality["total"]
    valid = quality["valid"] or 0
    invalid = total - valid

    print(f"Total records: {total:,}")
    print(f"Valid timestamps: {valid:,}")
    print(f"Invalid timestamps: {invalid:,}")

    if invalid > 0:
        print(
            f"WARNING: dropping {invalid:,} records "
            "with invalid timestamps."
        )

    # Keep columns needed
    df = (
        df
        .filter(F.col("event_time").isNotNull())
        .select(
            "user_id",
            "event_time",
            "artist_name",
            "track_id",
            "track_name",
        )
    )

    # Tie-breaker for records with exactly the same timestamp and track ID
    df = df.withColumn(
        "row_id",
        F.monotonically_increasing_id(),
    )

    return df


def mark_session_starts(df):

    ordering = [
        F.col("event_time"),
        F.col("track_id"),
        F.col("row_id"),
    ]

    user_window = (
        Window
        .partitionBy("user_id")
        .orderBy(*ordering)
    )

    df = df.withColumn(
        "previous_event_time",
        F.lag("event_time").over(user_window),
    )

    df = df.withColumn(
        "gap_seconds",
        F.col("event_time").cast("long")
        - F.col("previous_event_time").cast("long"),
    )

    df = df.withColumn(
        "new_session",
        F.when(
            F.col("previous_event_time").isNull()
            | (
                F.col("gap_seconds")
                > SESSION_GAP_SECONDS
            ),
            1,
        ).otherwise(0),
    )

    return df


def sessionize(df):

    print("Creating sessions...")

    df = mark_session_starts(df)

    ordering = [
        F.col("event_time"),
        F.col("track_id"),
        F.col("row_id"),
    ]

    cumulative_window = (
        Window
        .partitionBy("user_id")
        .orderBy(*ordering)
        .rowsBetween(
            Window.unboundedPreceding,
            Window.currentRow,
        )
    )

    df = df.withColumn(
        "session_number",
        F.sum("new_session").over(cumulative_window),
    )

    df = df.withColumn(
        "session_id",
        F.concat_ws(
            "_",
            F.col("user_id"),
            F.col("session_number").cast("string"),
        ),
    )

    return df