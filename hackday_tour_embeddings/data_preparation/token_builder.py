from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, collect_list, size, struct, udf
from pyspark.sql.types import ArrayType, StringType


def compute_narratives(events_df: DataFrame, min_narrative_size: int) -> DataFrame:
    # UDF wrapper to extract individual event information
    EVENT_COLUMNS: List[str] = ["tour_id", "location_id", "location_type", "event_name"]
    token_builder_udf = udf(
        lambda events: build_tokens(events), ArrayType(StringType())
    )

    events_df = (
        events_df.groupBy("platform", "locale_code", "visitor_id", "session_id")
        .agg(collect_list(struct(*EVENT_COLUMNS)).alias("events"))
        .withColumn("narrative", token_builder_udf(col("events")))
        .filter(size("narrative") >= min_narrative_size)
    )

    return events_df


def tokenize_from_adp_requests(data: dict) -> str:
    if data["tour_id"] is None:
        raise ValueError(
            "tour_id must not be null for ActivityDetailPageRequest events"
        )

    tour_id = int(data["tour_id"])
    return f"t{tour_id}"


def tokenize_from_dest_page_requests(data: dict) -> str:
    if data["location_type"] is None:
        raise ValueError(
            "location_type must not be null for DestinationPageRequest events"
        )
    if data["location_id"] is None:
        raise ValueError(
            "location_id must not be null for DestinationPageRequest events"
        )

    location_type = data["location_type"].lower()
    location_id = int(data["location_id"])
    return f"{location_type};l{location_id}"


def build_tokens(events) -> List[str]:
    tokens = []
    for data in events:
        if data["event_name"] == "ActivityDetailPageRequest":
            tokens.append(tokenize_from_adp_requests(data))
        elif data["event_name"] == "DestinationPageRequest":
            tokens.append(tokenize_from_dest_page_requests(data))
    return tokens
