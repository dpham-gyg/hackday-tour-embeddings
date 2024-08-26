import numpy as np
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
import torch

from hackday_tour_embeddings.data_preparation import token_builder

# NARRATIVES_PATH_ALL = "/mnt/data/duy.pham/hackdays-24-06/narratives/all"
# NARRATIVES_PATH_TRAIN = "/mnt/data/duy.pham/hackdays-24-06/narratives/train"
# NARRATIVES_PATH_TEST = "/mnt/data/duy.pham/hackdays-24-06/narratives/test"
# TOUR_INDEX_PATH = "/mnt/data/duy.pham/hackdays-24-06/tour-vocab/"

# NARRATIVES_PATH_ALL = "/mnt/data/duy.pham/hackdays-24-06/middle/narratives/all"
# NARRATIVES_PATH_TRAIN = "/mnt/data/duy.pham/hackdays-24-06/middle/narratives/train"
# NARRATIVES_PATH_TEST = "/mnt/data/duy.pham/hackdays-24-06/middle/narratives/test"
# TOUR_INDEX_PATH = "/mnt/data/duy.pham/hackdays-24-06/middle/tour-vocab/"

# NARRATIVES_PATH_ALL = "/mnt/data/duy.pham/hackdays-24-06/smaller/narratives/all"
# NARRATIVES_PATH_TRAIN = "/mnt/data/duy.pham/hackdays-24-06/smaller/narratives/train"
# NARRATIVES_PATH_TEST = "/mnt/data/duy.pham/hackdays-24-06/smaller/narratives/test"
# TOUR_INDEX_PATH = "/mnt/data/duy.pham/hackdays-24-06/smaller/tour-vocab/"

NARRATIVES_PATH_ALL = "/mnt/data/duy.pham/hackdays-24-06/tiny/narratives/all"
NARRATIVES_PATH_TRAIN = "/mnt/data/duy.pham/hackdays-24-06/tiny/narratives/train"
NARRATIVES_PATH_TEST = "/mnt/data/duy.pham/hackdays-24-06/tiny/narratives/test"
TOUR_INDEX_PATH = "/mnt/data/duy.pham/hackdays-24-06/tiny/tour-vocab/"

MIN_NARRATIVE_SIZE = 20
MAX_NARRATIVE_SIZE = 100
MIN_VISITORS_PER_TOUR = 500

BERT_EMB_SIZE = 768


def load_visitor_click_data(
    spark: SparkSession,
    start_date: str,
    end_date: str,
    load_location: bool = False,
):
    ADP_REQUEST_PATH = "/mnt/analytics/cleaned/v1/ActivityDetailPageRequest"
    DESTINATION_PAGE_REQUEST_PATH = "/mnt/analytics/cleaned/v1/DestinationPageRequest"

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


def split_visitor_clicks_to_train_test(visitor_clicks: DataFrame, test_ratio: float):
    visitors = visitor_clicks.select("visitor_id").distinct()
    tours = visitor_clicks.select("tour_id").distinct()

    train_visitors, test_visitors = visitors.randomSplit([1 - test_ratio, test_ratio])
    train_tours, test_tours = tours.randomSplit([1 - test_ratio, test_ratio])

    train_df = visitor_clicks.join(train_visitors, on="visitor_id").join(
        train_tours, on="tour_id"
    )
    test_df = visitor_clicks.join(test_visitors, on="visitor_id").join(
        test_tours, on="tour_id"
    )

    return train_df, test_df


def extract_tour_vocab_from_narratives(visitor_clicks: DataFrame):
    """
    visitor_clicks: DataFrame must have columns: visitor_id, tour_id
    """
    df = (
        visitor_clicks.select("tour_id")
        .dropDuplicates()
        .sort("tour_id")
        .withColumn("tour_token", F.concat_ws("", F.lit("t"), F.col("tour_id")))
        .withColumn("tour_index", F.monotonically_increasing_id() + 1)
        .select("tour_token", "tour_index")
    )

    w = Window.orderBy("tour_index")
    return df.withColumn("tour_index", F.row_number().over(w))


def load_tour_index_map(spark: SparkSession, path: str):
    """
    res = {}
    for f in files:
        with open(f) as f:
            for line in f:
                line = json.loads(line)
                res[line["tour_token"]] = line["tour_index"]
    return res
    """

    tour_indices = spark.read.json(path).toPandas().to_dict("records")
    return {x["tour_token"]: x["tour_index"] for x in tour_indices}


def load_tour_bert_embeddings(spark: SparkSession):
    raw_embs = (
        spark.table("gdp.sl_distilbert_keyword_tour_embeddings")
        .withColumn("tour_token", F.concat_ws("", F.lit("t"), F.col("tour_id")))
        .select("tour_token", "description_embeddings")
    )

    tour_indices = spark.read.json(TOUR_INDEX_PATH)

    embs_collected = (
        raw_embs.join(tour_indices, on="tour_token")
        .select("tour_index", "description_embeddings")
        .toPandas().to_dict("records")
    )

    return {x["tour_index"]: x["description_embeddings"] for x in embs_collected}


def prepare_bert_emb_tensor(bert_emb_matrix, tour_index_map):
    bert_default_emb = np.zeros(BERT_EMB_SIZE)

    bert_collected_embs = {0: bert_default_emb}
    for _, tour_index in tour_index_map.items():
        if tour_index not in bert_emb_matrix:
            bert_collected_embs[tour_index] = bert_default_emb
        else:
            bert_collected_embs[tour_index] = bert_emb_matrix[tour_index]

    sorted_embs =  sorted(bert_collected_embs.items(), key=lambda x: x[0])
    sorted_embs = [x[1] for x in sorted_embs]
    return torch.Tensor(sorted_embs)


def run(start_date, end_date, spark):
    print("Loading visitor click data")
    events = load_visitor_click_data(spark, start_date, end_date)
    print(
        f"there are {events.select('tour_id').dropDuplicates().count()} distinct tours"
    )

    w = Window.partitionBy("tour_id")
    events = events.withColumn(
        "n_visitors", F.size(F.collect_set(F.col("visitor_id")).over(w))
    )
    events = events.filter(F.col("n_visitors") >= MIN_VISITORS_PER_TOUR)

    print(
        f"there are {events.select('tour_id').dropDuplicates().count()} tours with at least {MIN_VISITORS_PER_TOUR} visitors"
    )
    print(f"there are {events.select('visitor_id').dropDuplicates().count()} visitors")

    tour_vocab = extract_tour_vocab_from_narratives(events)
    tour_vocab.write.mode("overwrite").json(TOUR_INDEX_PATH)
    print(f"tour vocab size = {tour_vocab.count()}, saved to {TOUR_INDEX_PATH}")

    train_events, test_events = split_visitor_clicks_to_train_test(
        events, test_ratio=0.1
    )

    train_narratives = token_builder.compute_narratives(
        train_events, min_narrative_size=MIN_NARRATIVE_SIZE
    )
    test_narratives = token_builder.compute_narratives(
        test_events, min_narrative_size=MIN_NARRATIVE_SIZE
    )
    all_narratives = token_builder.compute_narratives(
        events, min_narrative_size=MIN_NARRATIVE_SIZE
    )

    print(f"training set size = {train_narratives.count()}")
    print(f"test set size = {test_narratives.count()}")
    print(f"all narratives size = {all_narratives.count()}")

    train_narratives.write.mode("overwrite").json(NARRATIVES_PATH_TRAIN)
    test_narratives.write.mode("overwrite").json(NARRATIVES_PATH_TEST)
    all_narratives.write.mode("overwrite").json(NARRATIVES_PATH_ALL)
