from pyspark.sql.functions import col, when, concat_ws, expr, split, isnan, to_date


def preprocess_df(df):
    # filter rows without date or close value
    df = df.filter(~(col("date").isNull() | isnan(col("date"))))
    df = df.filter(~(col("close").isNull() | isnan(col("close"))))

    # create col date in DATE format
    df = df.withColumn("split_date", split(col("date"), "/"))
    df = df.drop("date")

    df = df.withColumn("month", when(col("split_date").getItem(0) < 10, expr("concat('0', split_date[0])"))
                       .otherwise(col("split_date").getItem(0).cast("string"))) \
        .withColumn("day",
                    when(col("split_date").getItem(1) < 10,
                         expr("concat('0', split_date[1])"))
                    .otherwise(col("split_date").getItem(1).cast("string"))) \
        .withColumn("year",
                    col("split_date").getItem(2).cast("string")) \
        .withColumn("date_temp",
                    expr("concat(year, '-', month, '-', day)")) \
        .drop("month", "day", "year")

    df = df.withColumn("date", to_date(col("date_temp")))
    df = df.drop("date_temp", "split_date")

    return df