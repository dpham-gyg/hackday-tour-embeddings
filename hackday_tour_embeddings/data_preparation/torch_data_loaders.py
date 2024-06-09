import json
from typing import List

import torch
from pyspark.sql import SparkSession
from torch.utils.data import DataLoader, Dataset, IterableDataset

from hackday_tour_embeddings.data_preparation import data_loading


def create_torch_train_test_data_loaders(
    narrative_files,
    tour_index_files,
    batch_size: int,
):
    """
    dataset = NarrativesDataset(spark, narrative_file, tour_index_file)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [0.9, 0.1])

    return DataLoader(train_ds, batch_size=64, shuffle=True), DataLoader(
        test_ds, batch_size=64, shuffle=False
    )
    """
    dataset = NarrativesDatasetIterable(narrative_files, tour_index_files)
    return DataLoader(dataset, batch_size=batch_size)


def list_dbfs_readable_files(path, dbutils) -> List[str]:
    return [
        x.path.replace("dbfs:/", "/dbfs/")
        for x in dbutils.fs.ls(path)
        if ".json" in x.path
    ]


def load_tour_bert_embeddings(spark: SparkSession):
    raw_embs = (
        spark.table("gdp.sl_distilbert_keyword_tour_embeddings")
        .withColumn("tour_token", F.concat_ws("", F.lit("t"), F.col("tour_id")))
        .select("tour_token", "description_embeddings")
    )

    tour_indices = spark.read.json(data_loading.TOUR_INDEX_PATH)

    return raw_embs.join(tour_indices, on="tour_token").select(
        "tour_index", "description_embeddings"
    )


class NarrativesDataset(Dataset):
    def __init__(self, spark: SparkSession, narrative_file: str, tour_index_file: str):
        self.tour_indices = data_loading.load_tour_index_map(spark, tour_index_file)

        """
        self.narratives= []
        for json_file in narrative_files:
            with open(json_file) as f:
                for sample_line in f:
                    narrative = json.loads(sample_line)["narrative"]
                    self.narratives.append(torch.Tensor(
                        [self.tour_indices[token] for token in narrative]
                    ))
        """
        narratives = spark.read.json(narrative_file).toPandas().to_dict("records")
        self.narratives = [
            torch.Tensor([self.tour_indices[token] for token in x["narrative"]])
            for x in narratives
        ]

    def __len__(self):
        return len(self.narratives)

    def __getitem__(self, idx):
        return self.narratives[idx]


class NarrativesDatasetIterable(IterableDataset):
    def __init__(self, narrative_files: str, tour_index_files: str):
        self.tour_indices = {}
        for f in tour_index_files:
            with open(f) as f:
                for line in f:
                    line = json.loads(line)
                    self.tour_indices[line["tour_token"]] = line["tour_index"]

        self.narrative_files = narrative_files

    def __iter__(self):
        for json_file in self.narrative_files:
            with open(json_file) as f:
                for sample_line in f:
                    narrative = json.loads(sample_line)["narrative"]

                    narrative_tensor = torch.LongTensor(
                        [self.tour_indices[token] for token in narrative]
                    )

                    narrative_multi_hot = (
                        torch.nn.functional.one_hot(
                            narrative_tensor, num_classes=len(self.tour_indices) + 1
                        )
                        .max(dim=0)
                        .values
                        .to(torch.float32)
                    )

                    yield narrative_multi_hot
