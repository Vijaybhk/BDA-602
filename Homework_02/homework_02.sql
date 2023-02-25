-- Use the Baseball Database
USE baseball;

-- batter_counts table
-- SELECT batter, game_id, Hit, atBat FROM batter_counts ORDER BY batter, game_id LIMIT 0, 20;

-- Game Table
-- SELECT game_id, local_date, YEAR(local_date) AS game_year FROM game LIMIT 0, 20;


-- Creating Tables to store the calculated averages
-- Historic batting average for each player
DROP TABLE IF EXISTS batter_historic_stats;

CREATE TABLE IF NOT EXISTS batter_historic_stats AS
SELECT
    batter
    , SUM(Hit) AS historic_hits
    , SUM(atBat) AS historic_atBats
    , IF( SUM(atBat) = 0, NULL, SUM(Hit) / SUM(atBat)) AS historic_average
FROM batter_counts
GROUP BY batter
ORDER BY batter
;

ALTER TABLE batter_historic_stats ADD PRIMARY KEY (batter);

SELECT batter, historic_hits, historic_atBats, historic_average FROM batter_historic_stats ORDER BY batter LIMIT 0, 20;

-- Annual batting average for each player
DROP TABLE IF EXISTS batter_annual_stats;

CREATE TABLE IF NOT EXISTS batter_annual_stats AS
SELECT
    b.batter
    , YEAR(g.local_date) AS game_year
    , SUM(Hit) AS annual_hits
    , SUM(atBat) AS annual_atBats
    , IF(SUM(atBat) = 0, NULL, SUM(Hit) / SUM(atBat)) AS annual_average
FROM batter_counts b JOIN game g ON b.game_id = g.game_id
GROUP BY batter, game_year
ORDER BY batter, game_year
;

ALTER TABLE batter_annual_stats ADD PRIMARY KEY (batter, game_year);

SELECT batter, game_year, annual_hits, annual_atBats, annual_average FROM batter_annual_stats LIMIT 0, 20;

-- Rolling batting average for each player
DROP TABLE IF EXISTS batter_rolling_stats;

CREATE TABLE IF NOT EXISTS batter_rolling_stats AS
WITH dummy_table (batter, game_date, hits_per_day, atBats_per_day) AS (
    SELECT
        b.batter
        , DATE(g.local_date) AS game_date
        , SUM(Hit) AS hits_per_day
        , SUM(atBat) AS atBats_per_day
    FROM batter_counts b JOIN game g ON b.game_id = g.game_id
    GROUP BY batter, game_date  -- done to include games played by same player in a day
    ORDER BY batter, game_date
)
SELECT
    A.batter
    , A.game_date
    , SUM(B.hits_per_day) AS rolling_hits
    , SUM(B.atBats_per_day) AS rolling_atBats
    , IF(SUM(B.atBats_per_day) = 0, NULL, SUM(B.hits_per_day) / SUM(B.atBats_per_day)) AS rolling_average
FROM dummy_table A JOIN dummy_table B
    ON A.batter = B.batter
        AND A.game_date > B.game_date
        AND B.game_date >= DATE_SUB(A.game_date, INTERVAL 100 DAY)
GROUP BY A.batter, A.game_date
ORDER BY A.batter, A.game_date
;

ALTER TABLE batter_rolling_stats ADD PRIMARY KEY (batter, game_date);

SELECT batter, game_date, rolling_hits, rolling_atBats, rolling_average FROM batter_rolling_stats LIMIT 0, 20;
