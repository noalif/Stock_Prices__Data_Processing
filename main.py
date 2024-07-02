from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from data_preprocess import preprocess_df
from pyspark.sql.functions import (col, avg, lag, stddev, countDistinct, sqrt, year,
                                   rank, date_sub, row_number, datediff, round)



def top_three_return_dates(df):
    df_Q4 = df.withColumn("30_days_ago", date_sub(col("Date"), 30))

    # find the closest available date within 30 days
    df_Q4 = df_Q4.alias("df1").join(
        df_Q4.alias("df2"),
        (col("df1.ticker") == col("df2.ticker")) &
        (col("df2.date") <= col("df1.30_days_ago")),
        "left"
    ).withColumn(
        "date_diff", datediff(col("df1.30_days_ago"), col("df2.date"))
    ).withColumn(
        "row_num", row_number().over(Window.partitionBy("df1.ticker", "df1.date").orderBy(col("date_diff").asc()))
    ).filter(
        col("row_num") == 1
    ).select(
        col("df1.ticker").alias("ticker"),
        col("df1.date").alias("date"),
        col("df1.close").alias("current_close"),
        col("df2.date").alias("prev_Date"),
        col("df2.close").alias("prev_Close")
    )

    df_result = (df_Q4.withColumn("30_Daily_Return", (col("current_close") - col("prev_Close")) /
                                  col("prev_Close") * 100).filter(col("prev_Close").isNotNull()))

    windowSpec = Window.partitionBy("ticker").orderBy(col("30_Daily_Return").desc())
    df_ranked = df_result.withColumn("rank", rank().over(windowSpec))

    # get top three 30-day return dates
    top_three_dates = df_ranked.filter(col("rank") <= 3).select("ticker", "date")

    return top_three_dates


def most_volatile_stock(spark, df):
    # Filter rows where daily return can't be calculated (first row per stock)
    df_daily_return = df.filter(col("daily_return").isNotNull())

    df_daily_return = df_daily_return.withColumn("year", year("date"))
    # number of trading days for each stock per year:
    df_standard_dev_1 = df_daily_return.groupBy("ticker", "year").agg(
        countDistinct("date").alias("trading_days_per_year"))

    df_standard_dev_2 = df_daily_return.groupBy("ticker", "year").agg(
        stddev("daily_return").alias("std_dev_daily_return"))

    df_standard_dev = df_standard_dev_1.join(df_standard_dev_2, on=["ticker", "year"])

    df_standard_dev = df_standard_dev.withColumn("standard_deviation",
                                                 col("std_dev_daily_return") * sqrt(col("trading_days_per_year")))

    # most volatile stock:
    most_vol_stock = df_standard_dev.orderBy(col("standard_deviation").desc()).select("ticker", "standard_deviation").first()
    df_Q3_res = spark.createDataFrame([most_vol_stock])

    return df_Q3_res

def most_frequently_traded_stock(df):
    df_trading_frequency = df.withColumn("trading_frequency", col("close") * col("volume"))
    df_traded_stock = df_trading_frequency.groupBy("ticker").agg(avg("trading_frequency").alias("frequency"))
    most_traded_stock = df_traded_stock.orderBy(df_traded_stock["frequency"].desc()).limit(1)
    return most_traded_stock

def average_daily_return(df, windowSpec):
    # daily return for each stock:
    df_daily_return = df.withColumn("prev_close", lag("close").over(windowSpec))
    df_daily_return = df_daily_return.withColumn("daily_return", ((col("close") - col("prev_close")) /
                                                                  col("prev_close")) * 100)
    df_avg_daily_return = df_daily_return.groupBy("date").agg(avg("daily_return").alias("average_return"))

    # important comment:
    # Dates with a NULL value in the average_return column are the first recorded dates for each stock,
    # so when calculating the RETURN value we could not take into account the previous day's value.

    return df_daily_return, df_avg_daily_return


def main():
    spark = SparkSession.builder.appName("Stock_Prices_Processing").getOrCreate()
    file_path = "stock_prices.csv"
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    df = preprocess_df(df)

    windowSpec = Window.partitionBy("ticker").orderBy(col("date").asc())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Q1 - The average daily return of all stocks for every date
    df_daily_return, df_avg_daily_return = average_daily_return(df, windowSpec)
    df_avg_daily_return.show()
    df_avg_daily_return.write.csv("Q1_res.csv", mode="overwrite", header=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Q2 - most frequently traded stock - as measured by closing price * volume - on average
    most_freq_traded_stock = most_frequently_traded_stock(df)
    most_freq_traded_stock.write.csv("Q2_res.csv", mode="overwrite", header=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Q3 - most volatile stock - as measured by the annualized standard deviation of daily returns
    most_vol_stock = most_volatile_stock(spark, df_daily_return)
    most_vol_stock.write.csv("Q3_res.csv", mode="overwrite", header=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Q4 - The top three 30-day return dates — present ticker and date combinations
    top_three_dates = top_three_return_dates(df)
    top_three_dates.write.csv("Q4_res.csv", mode="overwrite", header=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    spark.stop()



if __name__ == "__main__":
    main()









