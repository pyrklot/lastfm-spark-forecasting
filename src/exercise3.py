import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, functions as F

from sessions import load_data, mark_session_starts


SHUFFLE_PARTITIONS = 32
SEASON_LENGTH = 7


def create_spark():

    return (
        SparkSession.builder
        .appName("LastFMExercise3")
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


def find_top_user(df):

    print("Finding user with the most sessions...")

    session_starts = mark_session_starts(df)

    session_counts = (
        session_starts
        .filter(F.col("new_session") == 1)
        .groupBy("user_id")
        .agg(
            F.count("*").alias("session_count")
        )
    )

    top_user = (
        session_counts
        .orderBy(
            F.col("session_count").desc(),
            F.col("user_id").asc(),
        )
        .first()
    )

    print(
        f"Top user: {top_user['user_id']}"
        f" ({top_user['session_count']:,} sessions)"
    )

    return top_user["user_id"]


def build_daily_session_counts(df, top_user):

    print("Building daily session counts...")

    user_df = (
        df
        .filter(F.col("user_id") == top_user)
    )

    session_starts = mark_session_starts(user_df)

    daily_counts = (
        session_starts
        .filter(F.col("new_session") == 1)
        .withColumn(
            "date",
            F.to_date("event_time"),
        )
        .groupBy("date")
        .agg(
            F.count("*").alias("sessions")
        )
    )

    all_dates = (
        user_df
        .select(
            F.to_date("event_time").alias("date")
        )
        .agg(
            F.min("date").alias("min_date"),
            F.max("date").alias("max_date"),
        )
        .select(
            F.explode(
                F.sequence(
                    F.col("min_date"),
                    F.col("max_date"),
                    F.expr("INTERVAL 1 DAY"),
                )
            ).alias("date")
        )
    )

    daily_counts = (
        all_dates
        .join(
            daily_counts,
            on="date",
            how="left",
        )
        .fillna(
            0,
            subset=["sessions"],
        )
        .orderBy("date")
    )

    return daily_counts


def seasonal_naive_forecast(
    train,
    horizon,
    season_length=7,
):

    if len(train) < season_length:
        raise ValueError(
            "Training data is shorter than the seasonal period."
        )

    last_values = (
        train["sessions"]
        .iloc[-season_length:]
        .values
    )

    forecast_values = np.resize(
        last_values,
        horizon,
    )

    return forecast_values

def calculate_mase(actual, predicted, training):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(
        np.abs(actual - predicted)
    )

    training_values = np.asarray(
        training["sessions"],
        dtype=float,
    )

    scale = np.mean(
        np.abs(
            training_values[SEASON_LENGTH:]
            - training_values[:-SEASON_LENGTH]
        )
    )

    if scale == 0:
        return np.nan

    return mae / scale


def analyze_residuals(actual, predicted, label):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    residuals = actual - predicted
    absolute_errors = np.abs(residuals)

    horizons = np.arange(
        1,
        len(actual) + 1,
    )

    residual_df = pd.DataFrame(
        {
            "horizon": horizons,
            "actual": actual,
            "predicted": predicted,
            "residual": residuals,
            "absolute_error": absolute_errors,
        }
    )

    print(f"\nResidual analysis: {label}")

    print("\nOverall residual statistics:")

    print(
        f"Mean residual: "
        f"{residual_df['residual'].mean():.4f}"
    )

    print(
        f"Median residual: "
        f"{residual_df['residual'].median():.4f}"
    )

    print(
        f"Residual std: "
        f"{residual_df['residual'].std():.4f}"
    )

    print(
        f"Mean absolute error: "
        f"{residual_df['absolute_error'].mean():.4f}"
    )

    print(
        "\nResidual statistics by forecast horizon:"
    )

    horizon_groups = [
        ("Days 1-7", 1, 7),
        ("Days 8-30", 8, 30),
        ("Days 31-60", 31, 60),
        ("Days 61-90", 61, 90),
    ]

    for group_name, start, end in horizon_groups:

        group = residual_df[
            (residual_df["horizon"] >= start)
            & (residual_df["horizon"] <= end)
        ]

        if len(group) == 0:
            continue

        print(
            f"{group_name}: "
            f"MAE={group['absolute_error'].mean():.4f}, "
            f"mean residual={group['residual'].mean():.4f}, "
            f"std={group['residual'].std():.4f}"
        )

    print(
        "\nResidual quantiles:"
    )

    for quantile in [0.10, 0.25, 0.50, 0.75, 0.90]:

        value = residual_df["residual"].quantile(
            quantile
        )

        print(
            f"{quantile:.0%}: {value:.4f}"
        )

    return residual_df



def rolling_origin_backtest(
    daily_series,
    start_origin,
    end_origin,
    horizon=90,
    step=30,
):

    results = []

    for origin in range(
        start_origin,
        end_origin + 1,
        step,
    ):

        train = daily_series.iloc[:origin]

        actual = daily_series.iloc[
            origin:origin + horizon
        ]

        if len(actual) < horizon:
            break

        predicted = seasonal_naive_forecast(
            train,
            horizon,
            season_length=SEASON_LENGTH,
        )

        residuals = (
            actual["sessions"].values
            - predicted
        )

        for horizon_number, residual in enumerate(
            residuals,
            start=1,
        ):

            results.append(
                {
                    "origin": origin,
                    "horizon": horizon_number,
                    "actual": actual["sessions"].iloc[
                        horizon_number - 1
                    ],
                    "predicted": predicted[
                        horizon_number - 1
                    ],
                    "residual": residual,
                    "absolute_error": abs(residual),
                }
            )

    results_df = pd.DataFrame(results)

    print("\nRolling-origin backtest:")
    print(
        f"Number of forecast origins: "
        f"{results_df['origin'].nunique()}"
    )
    print(
        f"Total forecast errors: "
        f"{len(results_df):,}"
    )

    print(
        "\nOverall rolling-origin performance:"
    )

    rolling_mae = results_df[
        "absolute_error"
    ].mean()

    rolling_rmse = np.sqrt(
        np.mean(
            results_df["residual"] ** 2
        )
    )

    rolling_bias = results_df[
        "residual"
    ].mean()

    print(
        f"MAE:  {rolling_mae:.4f}"
    )

    print(
        f"RMSE: {rolling_rmse:.4f}"
    )

    print(
        f"Bias: {rolling_bias:.4f}"
    )

    print(
        "\nRolling-origin error by forecast horizon:"
    )

    horizon_groups = [
        ("Days 1-7", 1, 7),
        ("Days 8-30", 8, 30),
        ("Days 31-60", 31, 60),
        ("Days 61-90", 61, 90),
    ]

    for group_name, start, end in horizon_groups:

        group = results_df[
            (results_df["horizon"] >= start)
            & (results_df["horizon"] <= end)
        ]

        print(
            f"{group_name}: "
            f"MAE={group['absolute_error'].mean():.4f}, "
            f"std={group['residual'].std():.4f}, "
            f"bias={group['residual'].mean():.4f}"
        )

    print(
        "\nRolling-origin residual quantiles by forecast horizon:"
    )

    for group_name, start, end in horizon_groups:

        group = results_df[
            (results_df["horizon"] >= start)
            & (results_df["horizon"] <= end)
        ]

        q10 = group["residual"].quantile(0.10)
        q25 = group["residual"].quantile(0.25)
        q50 = group["residual"].quantile(0.50)
        q75 = group["residual"].quantile(0.75)
        q90 = group["residual"].quantile(0.90)

        print(
            f"{group_name}: "
            f"q10={q10:.4f}, "
            f"q25={q25:.4f}, "
            f"q50={q50:.4f}, "
            f"q75={q75:.4f}, "
            f"q90={q90:.4f}"
        )

    print(
        "\nOverall rolling-origin residual quantiles:"
    )

    for quantile in [0.10, 0.25, 0.50, 0.75, 0.90]:

        value = results_df[
            "residual"
        ].quantile(quantile)

        print(
            f"{quantile:.0%}: {value:.4f}"
        )

    return results_df


def create_final_forecast(daily_pd, top_user, horizon=90):

    print("\nCreating final 90-day forecast...")

    forecast_values = seasonal_naive_forecast(
        daily_pd,
        horizon,
        season_length=SEASON_LENGTH,
    )

    last_date = daily_pd.index.max()

    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    forecast_df = pd.DataFrame(
        {
            "date": forecast_dates,
            "forecast_sessions": forecast_values,
        }
    )

    lower_residual = -4
    upper_residual = 3

    forecast_df["lower_80"] = np.maximum(
        0,
        forecast_df["forecast_sessions"]
        + lower_residual,
    )

    forecast_df["upper_80"] = (
        forecast_df["forecast_sessions"]
        + upper_residual
    )

    forecast_df["user_id"] = top_user


    forecast_df = forecast_df[
        [
            "user_id",
            "date",
            "forecast_sessions",
            "lower_80",
            "upper_80",
        ]
    ]

    print(
        f"Forecast period: "
        f"{forecast_dates.min().date()}"
        f" → "
        f"{forecast_dates.max().date()}"
    )

    print("\nFirst 20 forecast days:")

    print(
        forecast_df.head(20).to_string(
            index=False
        )
    )

    output_path = (
        "/app/output/exercise3_forecast.tsv"
    )

    forecast_df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        "\n90-day forecast summary:"
    )

    print(
        f"Total forecast sessions: "
        f"{forecast_df['forecast_sessions'].sum():.0f}"
    )

    print(
        f"Average sessions per day: "
        f"{forecast_df['forecast_sessions'].mean():.2f}"
    )

    print(
        f"Minimum sessions per day: "
        f"{forecast_df['forecast_sessions'].min():.0f}"
    )

    print(
        f"Maximum sessions per day: "
        f"{forecast_df['forecast_sessions'].max():.0f}"
    )

    print(
        f"Lower-bound total sessions: "
        f"{forecast_df['lower_80'].sum():.0f}"
    )

    print(
        f"Upper-bound total sessions: "
        f"{forecast_df['upper_80'].sum():.0f}"
    )

    print(
        f"\nForecast saved to: {output_path}"
    )

    return forecast_df


def main():
    spark = create_spark()

    try:
        df = load_data(spark)

        df = df.select(
            "user_id",
            "event_time",
            "track_id",
            "row_id",
        )

        top_user = find_top_user(df)

        daily_counts = build_daily_session_counts(
            df,
            top_user,
        )

        print("\nTime-series information:")

        daily_counts.select(
            F.min("date").alias("first_date"),
            F.max("date").alias("last_date"),
            F.count("*").alias("number_of_days"),
            F.avg("sessions").alias("mean_sessions"),
            F.min("sessions").alias("min_sessions"),
            F.max("sessions").alias("max_sessions"),
        ).show()

        print("\nDaily session counts:")

        daily_counts.show(
            20,
            truncate=False,
        )

        daily_pd = (
            daily_counts
            .toPandas()
            .sort_values("date")
        )

        daily_pd["date"] = (
            daily_pd["date"]
            .astype("datetime64[ns]")
        )

        daily_pd = daily_pd.set_index("date")

        validation_days = 90
        test_days = 90

        total_days = len(daily_pd)

        train_days = (
            total_days
            - validation_days
            - test_days
        )

        print("\nTrain / validation / test split:")
        print(f"Total days:      {total_days}")
        print(f"Training days:   {train_days}")
        print(f"Validation days: {validation_days}")
        print(f"Test days:       {test_days}")

        train = daily_pd.iloc[:train_days]

        validation = daily_pd.iloc[
            train_days:train_days + validation_days
        ]

        test = daily_pd.iloc[
            train_days + validation_days:
        ]

        print("\nDate ranges:")

        print(
            f"Training:   {train.index.min().date()}"
            f" → {train.index.max().date()}"
        )

        print(
            f"Validation: {validation.index.min().date()}"
            f" → {validation.index.max().date()}"
        )

        print(
            f"Test:       {test.index.min().date()}"
            f" → {test.index.max().date()}"
        )

        validation_predictions = seasonal_naive_forecast(
            train,
            len(validation),
            season_length=7,
        )

        validation_actual = validation["sessions"].values

        validation_mae = np.mean(
            np.abs(
                validation_actual
                - validation_predictions
            )
        )

        validation_rmse = np.sqrt(
            np.mean(
                (
                    validation_actual
                    - validation_predictions
                ) ** 2
            )
        )

        validation_bias = np.mean(
            validation_predictions
            - validation_actual
        )


        validation_mase = calculate_mase(
            validation_actual,
            validation_predictions,
            train
        )

        print("\nSeasonal-naïve validation results:")
        print(f"MAE:  {validation_mae:.4f}")
        print(f"RMSE: {validation_rmse:.4f}")
        print(f"Bias: {validation_bias:.4f}")
        print(f"MASE:  {validation_mase:.4f}")


        validation_residuals = analyze_residuals(
            validation_actual,
            validation_predictions,
            "validation",
        )

        # Seasonal-naïve test forecast
        test_history = daily_pd.iloc[
            :train_days + validation_days
        ]

        test_predictions = seasonal_naive_forecast(
            test_history,
            len(test),
            season_length=7,
        )

        test_actual = test["sessions"].values

        test_mae = np.mean(
            np.abs(
                test_actual
                - test_predictions
            )
        )

        test_rmse = np.sqrt(
            np.mean(
                (
                    test_actual
                    - test_predictions
                ) ** 2
            )
        )

        test_bias = np.mean(
            test_predictions
            - test_actual
        )

        test_mase = calculate_mase(
            test_actual,
            test_predictions,
            test_history,
        )

        print("\nSeasonal-naïve test results:")
        print(f"MAE:  {test_mae:.4f}")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"Bias: {test_bias:.4f}")
        print(f"MASE:  {test_mase:.4f}")


        test_residuals = analyze_residuals(
            test_actual,
            test_predictions,
            "test",
        )

        rolling_results = rolling_origin_backtest(
            daily_pd.iloc[:train_days],
            start_origin=365,
            end_origin=train_days - 90,
            horizon=90,
            step=30,
        )

        final_forecast = create_final_forecast(
            daily_pd,
            top_user,
            horizon=90,
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()