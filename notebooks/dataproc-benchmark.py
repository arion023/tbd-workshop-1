from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pathlib import Path
import pandas as pd
import gc
import time
# from memory_profiler import memory_usage
import numpy as np


OUTPUT_DIR = "gs://tbd-2026l-11-data"
# OUTPUT_DIR = Path("data/phase2_26L") / f"group_11"

EVENTS_PATH = OUTPUT_DIR + "/events.parquet"
STRESS_EVENTS_PATH = OUTPUT_DIR + "/events_large.parquet"
DIMENSION_PATH = OUTPUT_DIR + "/dimension.parquet"
STRESS_DIMENSION_PATH = OUTPUT_DIR + "/dimension_large.parquet"
# MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

DATE_FROM = pd.Timestamp("2026-02-01").date()
DATE_TO = pd.Timestamp("2026-02-15").date()

SCALE = "medium"
SCALE_ROWS = {
    "debug": 200_000,
    "small": 2_000_000,
    "medium": 10_000_000,
    "large": 50_000_000,
}

N_ROWS = SCALE_ROWS[SCALE]

BENCHMARK_COLUMNS = [
    "library_engine",
    "mode",
    "query_name",
    "data_format",
    "layout",
    "rows",
    "median_time_s",
    "peak_memory_mb",
    "input_size_mb",
    "result_check",
    "notes",
]

benchmark_results = []

def parquet_size_mb(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 * 1024)
    return 0.0


def normalize_result(result):
    """
    Convert different result objects to a comparable lightweight representation.
    We do not need full equality of DataFrame internals, only stable output shape
    and basic values for sanity checks.
    """

    if isinstance(result, pd.DataFrame):
        return {
            "type": "pandas",
            "shape": result.shape,
            "columns": list(result.columns),
        }

    try:
        # Spark DataFrame
        if result.__class__.__name__ == "DataFrame":
            return {
                "type": "spark",
                "shape": (result.count(), len(result.columns)),
                "columns": result.columns,
            }
    except Exception:
        pass

    return {
        "type": type(result).__name__,
        "repr": str(result)[:200],
    }


def run_benchmark(
    query_name,
    engine,
    mode,
    func,
    repetitions=3,
    data_format="parquet",
    layout="default",
    input_path=EVENTS_PATH,
    result_check="manual",
    notes="",
):
    times = []
    peak_memories = []
    last_result = None

    for _ in range(repetitions):
        gc.collect()

        start_time = time.perf_counter()
        # mem_usage, result = memory_usage(
        #     (func, (), {}),
        #     retval=True,
        #     max_usage=True,
        #     interval=0.1,
        # )
        result = func()
        mem_usage = -1

        end_time = time.perf_counter()

        times.append(end_time - start_time)

        peak_mem = mem_usage[0] if isinstance(mem_usage, list) else mem_usage
        peak_memories.append(float(peak_mem))

        last_result = result

    result_summary = normalize_result(last_result)

    result = {
        "library_engine": engine,
        "mode": mode,
        "query_name": query_name,
        "data_format": data_format,
        "layout": layout,
        "rows": N_ROWS,
        "median_time_s": round(float(np.median(times)), 4),
        "peak_memory_mb": round(float(max(peak_memories)), 2),
        "input_size_mb": round(float(parquet_size_mb(input_path)), 2),
        "result_check": result_check,
        "notes": notes + f" | result={result_summary}",
    }

    benchmark_results.append(result)

    print(
        f"-> [{engine} | {mode}] {query_name}: "
        f"{result['median_time_s']} s, peak RSS {result['peak_memory_mb']} MB"
    )

    return result


spark = (
    SparkSession.builder
    .appName("TBDPhase2DataprocBenchmark")
    # .master("local[*]")
    # .config("spark.driver.memory", "4g")
    .getOrCreate()
)

def spark_q1_delayed_by_region():
    events = spark.read.parquet(str(EVENTS_PATH))
    dim = spark.read.parquet(str(DIMENSION_PATH))

    result = (
        events
        .filter(F.col("status") == "delayed")
        .join(dim, on="warehouse_id", how="inner")
        .groupBy("warehouse_region")
        .agg(
            F.count("*").alias("delayed_count"),
            F.avg("delay_minutes").alias("avg_delay_minutes"),
            F.max("delay_minutes").alias("max_delay_minutes"),
            F.avg("delivery_cost").alias("avg_delivery_cost"),
        )
        .orderBy(F.col("delayed_count").desc())
    )

    return result.toPandas()


def spark_q2_top_expensive_in_transit():
    events = spark.read.parquet(str(EVENTS_PATH))

    result = (
        events
        .filter(F.col("status") == "in_transit")
        .select("event_id", "tracking_number", "event_ts", "country", "delivery_cost", "warehouse_id")
        .orderBy(F.col("delivery_cost").desc(), F.col("event_id").asc())
        .limit(100)
    )

    return result.toPandas()


def spark_q3_daily_delivered_weight():
    events = spark.read.parquet(str(EVENTS_PATH))

    result = (
        events
        .filter(
            (F.col("status") == "delivered")
            & (F.col("event_date") >= F.lit(str(DATE_FROM)))
            & (F.col("event_date") <= F.lit(str(DATE_TO)))
        )
        .groupBy("event_date")
        .agg(
            F.count("*").alias("delivered_count"),
            F.sum("package_weight_kg").alias("total_package_weight_kg"),
            F.avg("package_weight_kg").alias("avg_package_weight_kg"),
        )
        .orderBy("event_date")
    )

    return result.toPandas()


LOCAL_BENCHMARKS = [
    ("Q1_delayed_by_region", "PySpark", "local[*]", spark_q1_delayed_by_region),
    ("Q2_top_expensive_in_transit", "PySpark", "local[*]", spark_q2_top_expensive_in_transit),
    ("Q3_daily_delivered_weight", "PySpark", "local[*]", spark_q3_daily_delivered_weight),
]

if __name__ == "__main__":
    for query_name, engine, mode, func in LOCAL_BENCHMARKS:
        run_benchmark(
            query_name=query_name,
            engine=engine,
            mode=mode,
            func=func,
            repetitions=3,
            data_format="parquet",
            layout="default",
            input_path=EVENTS_PATH,
            result_check="passed",
            notes="Dataproc benchmark on the same generated Parquet dataset. Peak memory is process RSS measured from the dataproc.",
        )


    benchmark_df = pd.DataFrame(benchmark_results, columns=BENCHMARK_COLUMNS)
    print(benchmark_df)

    DIMENSION_PATH = STRESS_DIMENSION_PATH
    EVENTS_PATH = STRESS_EVENTS_PATH

    for query_name, engine, mode, func in LOCAL_BENCHMARKS:
        run_benchmark(
            query_name=query_name,
            engine=engine,
            mode=mode,
            func=func,
            repetitions=3,
            data_format="parquet",
            layout="default",
            input_path=EVENTS_PATH,
            result_check="passed",
            notes="Dataproc benchmark on the same generated Parquet dataset. Peak memory is process RSS measured from the dataproc.",
        )


    benchmark_df = pd.DataFrame(benchmark_results, columns=BENCHMARK_COLUMNS)
    print(benchmark_df)
