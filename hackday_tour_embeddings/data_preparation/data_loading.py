from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ADP_REQUEST_PATH = "/mnt/analytics/cleaned/v1/ActivityDetailPageRequest"
DESTINATION_PAGE_REQUEST_PATH = "/mnt/analytics/cleaned/v1/DestinationPageRequest"
BOOKING_PATH = "/mnt/analytics/cleaned/v1/BookAction"


def load_visitor_click_data(
    spark: SparkSession,
    start_date: str,
    end_date: str,
    load_location: bool = False,
):
    BASE_COLUMNS = [
        F.col("event_properties.timestamp").alias("timestamp"),
        F.col("header.platform").alias("platform"),
        F.col("user.visitor_id").alias("visitor_id"),
        F.col("user.session_id").alias("session_id"),
        F.col("user.locale_code").alias("locale_code"),
        F.col("date"),
        F.col("event_name"),
    ]

    adp_request_df = (
        spark.read.options(mergeSchema="true")
        .parquet(ADP_REQUEST_PATH)
        .select(BASE_COLUMNS + [F.col("tour_id")])
        .filter(F.col("date").between(start_date, end_date))
        .filter(F.col("tour_id").isNotNull())
        .filter(F.col("tour_id") > 0)
    )

    if not load_location:
        return adp_request_df.withColumn("location_type", F.lit(None)).withColumn(
            "location_id", F.lit(None)
        )

    destination_page_request_df = (
        spark.read.options(mergeSchema="true")
        .parquet(DESTINATION_PAGE_REQUEST_PATH)
        .select(BASE_COLUMNS + [F.col("location_type"), F.col("location_id")])
        .filter(F.col("location_type").isNotNull())
        .filter(F.col("location_id").isNotNull())
    )

    return adp_request_df.unionByName(
        destination_page_request_df, allowMissingColumns=True
    )


def load_visitor_booking_data(
    spark: SparkSession,
    start_date: str,
    end_date: str,
):
    return (
        spark.read.options(mergeSchema="true")
        .parquet(BOOKING_PATH)
        .filter(F.col("date").between(start_date, end_date))
        .select(
            "user.visitor_id",
            "pageview_properties.tour_ids",
            "event_properties.timestamp",
            "date",
        )
    )
