-- Use the Baseball Database
USE baseball;

-- batter_counts table
SELECT batter, game_id, Hit, atBat FROM batter_counts ORDER BY batter, game_id LIMIT 0, 20;

-- Game Table
SELECT game_id, local_date, YEAR(local_date) AS game_year FROM game LIMIT 0, 20;


-- Creating Tables to store the calculated averages
-- Historic batting average for each player
DROP TABLE IF EXISTS historic_stats;

CREATE TABLE IF NOT EXISTS historic_stats AS
SELECT
    batter, SUM(Hit) AS historic_hits, SUM(atBat) AS historic_atBats, IF( SUM(atBat) = 0, NULL, ROUND( SUM(Hit) / SUM(atBat), 3 )) AS historic_average
FROM batter_counts
GROUP BY batter
ORDER BY batter
;

ALTER TABLE historic_stats ADD PRIMARY KEY (batter);

SELECT batter, historic_hits, historic_atBats, historic_average FROM historic_stats ORDER BY batter LIMIT 0, 20;

-- Annual batting average for each player
DROP TABLE IF EXISTS annual_stats;

CREATE TABLE IF NOT EXISTS annual_stats AS
SELECT
    b.batter, YEAR(g.local_date) AS game_year, SUM(Hit) AS annual_hits, SUM(atBat) AS annual_atBats, IF(atBat = 0, NULL, ROUND(SUM(Hit) / SUM(atBat), 3)) AS annual_average
FROM batter_counts b JOIN game g ON b.game_id = g.game_id GROUP BY batter, game_year
;

ALTER TABLE annual_stats ADD PRIMARY KEY (batter, game_year);

SELECT batter, game_year, annual_hits, annual_atBats, annual_average FROM annual_stats LIMIT 0, 20;
