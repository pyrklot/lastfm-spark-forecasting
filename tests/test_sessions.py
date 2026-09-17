from pyspark.sql import SparkSession, functions as F

from sessions import sessionize


def create_test_spark():
    return (
        SparkSession.builder
        .appName("TestSessions")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def test_session_boundary():
    spark = create_test_spark()

    try:
        data = [
            ("user_1", "2025-01-01 10:00:00", "track_a"),
            ("user_1", "2025-01-01 10:10:00", "track_b"),
            ("user_1", "2025-01-01 10:30:00", "track_c"),
            ("user_1", "2025-01-01 10:50:01", "track_d"),
        ]

        df = (
            spark.createDataFrame(
                data,
                ["user_id", "event_time", "track_id"],
            )
            .withColumn(
                "event_time",
                F.to_timestamp("event_time"),
            )
            .withColumn(
                "artist_name",
                F.lit("test_artist"),
            )
            .withColumn(
                "track_name",
                F.col("track_id"),
            )
            .withColumn(
                "row_id",
                F.monotonically_increasing_id(),
            )
        )

        result = (
            sessionize(df)
            .select(
                "track_id",
                "session_number",
            )
            .orderBy("track_id")
            .collect()
        )

        session_numbers = [
            row["session_number"]
            for row in result
        ]

        assert session_numbers == [1, 1, 1, 2]

    finally:
        spark.stop()