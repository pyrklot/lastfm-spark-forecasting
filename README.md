# Last.fm Spark & Forecasting Challenge

This project contains solutions for the Last.fm 1K dataset coding and forecasting exercises.

The solution is implemented with PySpark and Python and runs inside Docker for a reproducible development environment.



## Project structure

```text
.
├── Dockerfile
├── requirements.txt
├── data/
│   └── .gitkeep
├── output/
│   ├── .gitkeep
│   ├── exercise2.tsv
│   └── exercise3_forecast.tsv
├── src/
│   ├── exercise2.py
│   ├── exercise3.py
│   ├── main.py
│   └── sessions.py
└── tests/
    └── test_sessions.py   
    

## Dataset

The project uses the Last.fm 1K dataset. 

The main play-history file is: 
userid-timestamp-artid-artname-traid-traname.tsv

The dataset is expected to be available under: 
data/lastfm-dataset-1K/

The dataset is not included in this repository. 



## Requirements

The project uses:

- Docker
- PySpark
- Python 3.11
- Java 17
- pandas
- NumPy
- scikit-learn
- statsmodels
- pytest
- JupyterLab

Docker usage allows Spark, Java and Python dependencies are consistent across environments.


## Build the Docker image
From the project root: docker build -t lastfm-spark .



## Exercise 2 

Assignment
Question: What are the top 10 songs played in the top 50 longest sessions by tracks count?
In this assignment, a user "session" consists of one or more songs played by a given user, where each song is started within 20 minutes of the previous song's start time.


Run: 
MSYS_NO_PATHCONV=1 docker run --rm \
  -e PYTHONPATH=/app/src \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/output:/app/output" \
  lastfm-spark \
  python /app/src/exercise2.py
  

The result is written to: 
output/exercise2.tsv


The output contains: 
artist_name
track_name
play_count


Implementation notes
The solution:
1. Loads the Last.fm play history with Spark.
2. Parses timestamps explicitly.
3. Orders events by user and event time.
4. Identifies session boundaries using the 20-minute rule.
5. Creates a session identifier using a cumulative window.
6. Finds the 50 sessions with the largest number of tracks.
7. Restricts the data to those sessions.
8. Counts songs by artist and track.
9. Returns the top 10 songs.

A deterministic tie-breaker is included for events with identical timestamps.

The implementation avoids caching the complete dataset because the full dataset is large and unnecessary caching can cause excessive JVM memory usage.



## Exercise 3 

Assignment
Select the top 1 user who has the highest number of sessions. 
Forecast the next 3 months of your selected metric, starting from the last available record for that user


Selected user
The user with the highest number of session is: user_000833
with: 6,897 sessions


Time series
The session counts are aggregated by day.

The available daily history contains:
1,554 days
2005-02-20 to 2009-05-23


Forecasting approach 
The final model uses a seasonal-naive forecasting approach with a 7-day seasonal period.

Each forecast value is based on the corresponding day of the previous week.

This approach was selected for: 
- simplicity
- transparency
- easy to reproduce
- appropriateness for capturing a weekly pattern
- inexpensive to operationalize

The model was evaluated using time-ordered validation and test periods, rather than random train/test splitting.


Validation
The data was split into: 
Training:   1,374 days
Validation:    90 days
Test:          90 days

Seasonal-naive validation metrics:
MAE:   2.5444
RMSE:  3.3183
MASE:  1.2044

test metrics:
MAE:   1.8333
RMSE:  2.3214
MASE:  0.8542

A rolling-origin backtest was also performed to evaluate performance across multiple historical forecast origins.

Rolling-origin results:
Origins: 31
Forecast errors: 2,790
MAE:  2.2090
RMSE: 2.9828


Final forecast
The  final model is trained using all available historical observations. 

The forecast period: 
2009-05-24 to 2009-08-21

The point forecast is: 
Total forecast sessions: 400
Average sessions/day:    4.44

The forecast is written to: 
output/exercise3_forecast.tsv

The output contains: 
user_id
date
forecast_sessions
lower_80
upper_80


Uncertainty estimation
Seasonal-naive does not provide a model-based predictive distribution.

Therefore, uncertainty was estimated empirically using residuals from the rolling-origin backtest.

The rolling-origin backtest retained horizon-specific errors from day 1 through day 90. 
Residual quantiles were examined across four forecast-horizon ranges and were broadly stable, with no systematic widening at longer horizons.

The empirical daily prediction band uses residual quantiles of approximately:
Lower residual: -4 sessions
Upper residual: +3 sessions

The same residual bounds are applied across the 90-day forecast horizon.

The lower bound is clipped at zero because session counts cannot be negative.

The empirical prediction band is based on 31 rolling-origin folds and 2,790 horizon-specific forecast errors. 

Because these errors are not fully independent, the tail quantiles should be treated as approximate.

These are empirical daily prediction bands, not a conventional model-based confidence interval.

The lower and upper daily bounds should not be interpreted as an 80% confidence interval for the total number of sessions across the full 90-day period.


Testing
Run the automated tests with: 
MSYS_NO_PATHCONV=1 docker run --rm \
  -e PYTHONPATH=/app/src \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/tests:/app/tests" \
  lastfm-spark \
  pytest -q /app/tests
  
Current test status: 
1 passed 


Performance considerations
The Last.fm dataset contains approximately 19 million play records. 

The main computationally expensive operation is sessionization as it requires ordering events within eacḥ user and applying window functions. 

The implementation includes several measures to reduce unnecessary memory usage: 
- Only required columns are retained
- Timestamp validation is performed with a single aggregation
- The complete dataset is not cached
- The top 50 sessions are a very small result and can be efficiently joined back to the event data 
- Adaptive query execution is enabled
- Shuffle partitions are configured for the local Docker environment
- The final 10 row result is written as a single TSV file


Assumptions
- A session belongs to one user
- A new session starts when the time between consecutive tracks exceeds 20 minutes
- Events exactly 20 minutes apart remain in the same session
- Invalid timestamps are excluded
- Events with identical timestamps are given a deterministic ordering using track ID and an internal row identifier
- A song is identified by a combination of artist name and track name
- Daily session counts are used for forecasting
- The forecasting horizon is 90 days to represent approximately three months


Possible improvements
With more time, the solution could be extended with:
- Additional data-quality checks
- More automated tests for edge cases in sessionization
- Additional forecasting baselines and models
- Holiday or calendar effects where appropriate
- Automated model selection 
- Model and data versioning
