# mariadb connection
# Source: https://kontext.tech/article/1061/pyspark-read-data-from-mariadb-database

import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession


class BaseballSpark:
    def __init__(
        self, database="baseball", master="local[*]", appname="PySpark - MariaDB"
    ):
        self.database = database
        self.master = master
        self.appname = appname

    def create_spark_session(self):
        spark = (
            SparkSession.builder.appName(self.appname).master(self.master).getOrCreate()
        )
        return spark

    def read_query_from_mariadb(
        self,
        session,
        sql_query,
        user="root",
        password="x11docker",  # pragma: allowlist secret
    ):

        server = "localhost"
        port = 3306
        jdbc_url = f"jdbc:mysql://{server}:{port}/{self.database}?permitMysqlScheme"
        jdbc_driver = "org.mariadb.jdbc.Driver"

        df = (
            session.read.format("jdbc")
            .option("url", jdbc_url)
            .option("query", sql_query)
            .option("user", user)
            .option("password", password)
            .option("driver", jdbc_driver)
            .load()
        )

        return df


def main():
    baseball_spark = BaseballSpark()
    batter_counts_query = """SELECT batter, game_id, Hit, atBat FROM batter_counts"""
    game_query = """SELECT game_id, DATE(local_date) as game_date FROM game"""

    spark = baseball_spark.create_spark_session()

    game_df = baseball_spark.read_query_from_mariadb(
        session=spark, sql_query=game_query
    )
    batter_counts_df = baseball_spark.read_query_from_mariadb(
        session=spark, sql_query=batter_counts_query
    )

    df = batter_counts_df.join(game_df, ["game_id"])
    df.createOrReplaceTempView("rolling_average")
    df.persist(StorageLevel.DISK_ONLY)

    df = spark.sql(
        """
            SELECT
                A.batter
                , A.game_date
                , SUM(B.Hit) as rolling_hits
                , SUM(B.atBat) as rolling_atBats
                , IF(SUM(B.atBat) = 0
                    , NULL
                    , SUM(B.Hit) / SUM(B.atBat)) AS rolling_average
            FROM rolling_average A join rolling_average B
                ON A.batter = B.batter
                    AND A.game_date > B.game_date
                    AND B.game_date >= DATE_SUB(A.game_date, 100)
            GROUP BY A.batter, A.game_date
            ORDER BY A.batter, A.game_date
        """
    )

    df.show()

    return


if __name__ == "__main__":
    sys.exit(main())
