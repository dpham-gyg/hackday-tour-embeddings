from pyspark.ml.feature import Word2Vec

MIN_NARRATIVES_PER_VECTOR = 20
EMB_SIZE = 300


def train_w2v(train_ds):
    w2v_model = (
        Word2Vec(
            inputCol="narrative",
            outputCol="embedding",
            vectorSize=EMB_SIZE,
        )
        .setMinCount(MIN_NARRATIVES_PER_VECTOR)
        .fit(train_ds)
    )

    return w2v_model
