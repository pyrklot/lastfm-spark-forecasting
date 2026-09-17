import os
import shutil

from pyspark.sql import SparkSession, functions as F

from sessions import load_data, sessionize


OUTPUT_FILE = "/app/output/exercise2.tsv"

SHUFFLE_PARTITIONS = 32


def create_spark():

    return (
        SparkSession.builder
        .appName("LastFMExercise2")
        .master("local[*]")
        .config(
            "spark.sql.shuffle.partitions",
            SHUFFLE_PARTITIONS,
        )
        .config(
            "spark.sql.adaptive.enabled",
            "true",
        )
        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            "true",
        )
        .getOrCreate()
    )


def find_top_50_sessions(df):

    print("Finding top 50 longest sessions...")

    session_counts = (
        df
        .groupBy(
            "user_id",
            "session_id",
        )
        .agg(
            F.count("*").alias("track_count")
        )
    )

    return (
        session_counts
        .orderBy(
            F.col("track_count").desc(),
            F.col("user_id").asc(),
            F.col("session_id").asc(),
        )
        .limit(50)
    )


def find_top_10_songs(df, top_50_sessions):

    print("Finding top 10 songs...")

    top_50_tracks = (
        df
        .join(
            F.broadcast(
                top_50_sessions.select(
                    "user_id",
                    "session_id",
                )
            ),
            on=[
                "user_id",
                "session_id",
            ],
            how="inner",
        )
        .filter(F.col("artist_name").isNotNull())
        .filter(F.col("track_name").isNotNull())
    )

    return (
        top_50_tracks
        .groupBy(
            "artist_name",
            "track_name",
        )
        .agg(
            F.count("*").alias("play_count")
        )
        .orderBy(
            F.col("play_count").desc(),
            F.col("artist_name").asc(),
            F.col("track_name").asc(),
        )
        .limit(10)
    )


def save_result(result):

    print("\nFinal result:")

    result.show(
        10,
        truncate=False,
    )

    output_dir = OUTPUT_FILE + "_tmp"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    (
        result
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("sep", "\t")
        .option("header", "true")
        .csv(output_dir)
    )

    part_files = [
        filename
        for filename in os.listdir(output_dir)
        if filename.startswith("part-")
        and filename.endswith(".csv")
    ]

    if len(part_files) != 1:
        raise RuntimeError(
            f"Expected exactly one Spark output file, "
            f"found {len(part_files)}"
        )

    shutil.move(
        os.path.join(output_dir, part_files[0]),
        OUTPUT_FILE,
    )

    shutil.rmtree(output_dir)

    print(f"\nResult written to: {OUTPUT_FILE}")


def main():
    spark = create_spark()

    try:
        # Shared logic lives in sessions.py.
        df = load_data(spark)
        sessionized_df = sessionize(df)

        # Exercise 2-specific logic
        top_50_sessions = find_top_50_sessions(
            sessionized_df
        )

        top_10_songs = find_top_10_songs(
            sessionized_df,
            top_50_sessions,
        )

        save_result(top_10_songs)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()