# mariadb connection
# Source: https://kontext.tech/article/1061/pyspark-read-data-from-mariadb-database

# Custom Transformer
# Source: https://www.crowdstrike.com/blog/
# deep-dive-into-custom-spark-transformers-for-machine-learning-pipelines/

# sql queries are formatted differently to use inputs into the query strings
# Also to get around pycharm-highlighting for other language injections

import sys

from pyspark import StorageLevel, keyword_only
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.param.shared import Param, Params
from pyspark.sql import DataFrame, SparkSession


class BaseballSpark:
    """
    Custom class to connect pyspark to baseball database stored in MariaDB
    """

    def __init__(
        self, database="baseball", master="local[*]", appname="PySpark - MariaDB"
    ):
        """
        Constructor: set values for all parameters
        :param database: default database is "baseball"
        :param master: default is local[*] to run locally
        :param appname: application name to display, default is "PySpark - MariaDB"
        """

        self.database = database
        self.master = master
        self.appname = appname

    def create_spark_session(self) -> SparkSession:
        """
        Method to create a Spark Session
        :return: returns a spark session
        """

        spark = (
            SparkSession.builder.appName(self.appname).master(self.master).getOrCreate()
        )
        return spark

    def read_query_from_mariadb(
        self,
        session: SparkSession = None,
        sql_query: str = None,
        user: str = "root",
        password: str = "x11docker",  # pragma: allowlist secret
    ) -> DataFrame:

        """
        Method to get Data from Mariadb using sql query accessing baseball database.
        Connection will be done using mariadb jdbc driver.
        Change user and password if necessary.
        :param session: spark session, created using create_spark_session method
        :param sql_query: query required to get data from a specific table
        :param user: user of MariaDB, default set as root
        :param password: password for the MariaDB, default set as x11docker
        :return: a pyspark DataFrame object generated from the sql query
        """

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


class BatterRollingAverageTransformer(Transformer):
    """
    Custom Transformer Class for Batter Rolling Average in Baseball
    """

    # New parameters required to input into the transformer
    session = Param(Params._dummy(), "session", "Spark Session Input")

    table_view = Param(
        Params._dummy(), "table_view", "Spark Data Frame View Input for SQL query"
    )

    @keyword_only
    def __init__(self, session=None, table_view=None):
        """
        Constructor to set values for all parameters.
        set defaults for any custom variables
        :param session: Input Spark session in the transformer
        :param table_view: Input Dataframe View for transformer query
        """

        super(BatterRollingAverageTransformer, self).__init__()
        self._setDefault(session=None)
        self._setDefault(table_view=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, session=None, table_view=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # Methods to set and get the custom variables
    def setSession(self, new_session):
        return self.setParams(session=new_session)

    def set_table_view(self, new_table_view):
        return self.setParams(table_view=new_table_view)

    def getSession(self):
        return self.getOrDefault(self.session)

    def get_table_view(self):
        return self.getOrDefault(self.table_view)

    def _transform(self, dataset):
        """
        Overwrite the transform method
        :param dataset: Input Spark DataFrame
        :return: return the transformed spark DataFrame
        """

        session = self.getSession()
        table_view = self.get_table_view()

        dataset = session.sql(
            """
        SELECT
            A.batter
            , A.game_date
            , SUM(B.Hit) AS rolling_hits
            , SUM(B.atBat) AS rolling_atBats
            , IF(SUM(B.atBat) = 0
                , NULL
                , SUM(B.Hit) / SUM(B.atBat)) AS rolling_average
        """
            + "FROM {} A JOIN {} B".format(table_view, table_view)
            + """
            ON A.batter = B.batter
                AND A.game_date > B.game_date
                AND B.game_date >= DATE_SUB(A.game_date, 100)
        GROUP BY A.batter, A.game_date
        ORDER BY A.batter, A.game_date
        """
        )

        return dataset


def main():
    """
    Main function to get two required tables from baseball and
    batter's rolling average using custom transformer.
    """

    # Instantiate the BaseballSpark class created above
    baseball_spark = BaseballSpark()

    # Queries to get the game and batter_counts tables from baseball database
    batter_counts_query = "SELECT" + " batter, game_id, Hit, atBat FROM batter_counts"
    game_query = "SELECT" + " game_id, DATE(local_date) as game_date FROM game"

    # Get the Spark session
    spark = baseball_spark.create_spark_session()

    # Get the game dataframe from MariaDB connection
    game_df = baseball_spark.read_query_from_mariadb(
        session=spark, sql_query=game_query
    )

    # Get the batter_counts dataframe from MariaDB connection
    batter_counts_df = baseball_spark.read_query_from_mariadb(
        session=spark, sql_query=batter_counts_query
    )

    # Join the above two dataframes on game_id column
    df = batter_counts_df.join(game_df, ["game_id"])

    # Create a temporary view rolling_df used in transformer query
    df.createOrReplaceTempView("rolling_df")
    df.persist(StorageLevel.DISK_ONLY)

    # Create the transformer object
    rolling_transformer = BatterRollingAverageTransformer(
        session=spark, table_view="rolling_df"
    )

    # Using the pipeline to implement the transformer.
    # Used pipeline to make code easily readable and understandable
    # instead of calling transform method to return the dataframe directly
    pipe = Pipeline(stages=[rolling_transformer]).fit(df)
    df = pipe.transform(df)

    # Display the final results.
    # df.filter(df.batter == 110029).show()
    df.show()

    return


if __name__ == "__main__":
    sys.exit(main())
