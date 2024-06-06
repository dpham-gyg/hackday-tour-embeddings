import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ADP_REQUEST_PATH = "/mnt/analytics/cleaned/v1/ActivityDetailPageRequest"
DESTINATION_PAGE_REQUEST_PATH = "/mnt/analytics/cleaned/v1/DestinationPageRequest"


def load_visitor_click_data(spark: SparkSession, start_date: str, end_date: str):
    BASE_COLUMNS = [
        F.col("event_properties.timestamp").alias("timestamp"),
        F.col("header.platform").alias("platform"),
        F.col("user.visitor_id").alias("visitor_id"),
        F.col("user.session_id").alias("session_id"),
        F.col("user.locale_code").alias("locale_code"),
        F.col("date"),
        F.col("event_name"),
    ]

    adp_last_date = spark.read.parquet(
        os.path.join(ADP_REQUEST_PATH, f"date={end_date}")
    )
    adp_request_df = (
        spark.read.schema(adp_last_date.schema)
        .parquet(ADP_REQUEST_PATH)
        .select(BASE_COLUMNS + [F.col("tour_id")])
        .filter(F.col("date").between(start_date, end_date))
        .filter(F.col("tour_id").isNotNull())
        .filter(F.col("tour_id") > 0)
    )

    destination_page_last_date = spark.read.parquet(
        os.path.join(DESTINATION_PAGE_REQUEST_PATH, f"date={end_date}")
    )
    destination_page_request_df = (
        spark.read.schema(destination_page_last_date.schema)
        .parquet(DESTINATION_PAGE_REQUEST_PATH)
        .select(BASE_COLUMNS + [F.col("location_type"), F.col("location_id")])
        .filter(F.col("location_type").isNotNull())
        .filter(F.col("location_id").isNotNull())
    )

    return adp_request_df.unionByName(destination_page_request_df, allowMissingColumns=True)
