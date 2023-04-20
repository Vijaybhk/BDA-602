USE baseball;

CREATE TEMPORARY TABLE historic_start_pitcher ENGINE=MEMORY AS  -- noqa: PRS
WITH dummy_table (game_id, game_date, pitcher, startingPitcher, IP,
       homeTeam, awayTeam, HR, H, BB, BF, IBB, HBP, SO, PT, PTB, Winner) AS (
    SELECT
        pc.game_id
        , DATE(g.local_date)
        , pc.pitcher
        , pc.startingPitcher
        , pc.outsPlayed/3
        , pc.homeTeam
        , pc.awayTeam
        , pc.Home_Run
        , pc.Hit
        , pc.Walk
        , pc.plateApperance
        , pc.Intent_Walk
        , pc.Hit_By_Pitch
        , pc.Strikeout
        , pc.pitchesThrown
        , 0.89*(1.255*(pc.Hit-pc.Home_Run) + 4*pc.Home_Run)
              + 0.56*(pc.Walk+pc.Hit_By_Pitch-pc.Intent_Walk) AS PTB
        , bs.winner_home_or_away
    FROM game g
        JOIN pitcher_counts pc on g.game_id = pc.game_id
        JOIN boxscore bs on g.game_id = bs.game_id
    WHERE pc.startingPitcher=1
    GROUP BY pc.game_id, pc.pitcher
    ORDER BY pc.game_id
)
SELECT
    A.game_id
    , A.game_date
    , A.pitcher
    , A.homeTeam
    , SUM(B.BF) AS BFP
    , SUM(B.IP) AS IP
    , SUM(B.BB)/9 AS BB9
    , SUM(B.H)/9 AS HA9
    , SUM(B.HR)/9 AS HRA9
    , SUM(B.SO)/(9*SUM(B.IP)) AS SO9
    , SUM(B.SO)/SUM(B.PT) AS SOPercent
    , IF(SUM(B.IP) = 0
        , 0
        , 3 + ((13*SUM(B.HR)+3*(SUM(B.BB+B.HBP)-2*B.SO))/SUM(B.IP))
        ) AS DICE
    , IF(SUM(B.IP) = 0
        , 0
        , (SUM(B.H)+SUM(B.BB)) / SUM(B.IP)) AS WHIP
    , IF(SUM(B.IP)+SUM(B.BF) = 0
        , 0
        , 9*((SUM(B.H)+SUM(B.BB)+SUM(B.HBP))* SUM(B.PTB))/(SUM(B.BF)*SUM(B.IP))
              -0.56) AS CERA
    , IF(A.Winner='H', 1, 0) AS HTWins
FROM dummy_table A LEFT JOIN dummy_table B
    ON A.pitcher = B.pitcher
        AND A.game_date > B.game_date
GROUP BY A.pitcher, A.game_date
ORDER BY A.game_id, A.game_date, A.pitcher
;

ALTER TABLE historic_start_pitcher ADD PRIMARY KEY (game_id, pitcher);


CREATE TEMPORARY TABLE rolling_start_pitcher ENGINE=MEMORY AS   -- noqa: PRS
WITH dummy_table (game_id, game_date, pitcher, startingPitcher, IP,
       homeTeam, awayTeam, HR, H, BB, BF, IBB, HBP, SO, PT, PTB) AS (
    SELECT
        pc.game_id
        , DATE(g.local_date)
        , pc.pitcher
        , pc.startingPitcher
        , pc.outsPlayed/3
        , pc.homeTeam
        , pc.awayTeam
        , pc.Home_Run
        , pc.Hit
        , pc.Walk
        , pc.plateApperance
        , pc.Intent_Walk
        , pc.Hit_By_Pitch
        , pc.Strikeout
        , pc.pitchesThrown
        , 0.89*(1.255*(pc.Hit-pc.Home_Run) + 4*pc.Home_Run)
              + 0.56*(pc.Walk+pc.Hit_By_Pitch-pc.Intent_Walk) AS PTB
    FROM game g
        JOIN pitcher_counts pc on g.game_id = pc.game_id
    WHERE pc.startingPitcher=1
    GROUP BY pc.game_id, pc.pitcher
    ORDER BY pc.game_id
)
SELECT
    A.game_id
    , A.game_date
    , A.pitcher
    , A.homeTeam
    , SUM(B.BF) AS BFP
    , SUM(B.IP) AS IP
    , SUM(B.BB)/9 AS BB9
    , SUM(B.H)/9 AS HA9
    , SUM(B.HR)/9 AS HRA9
    , SUM(B.SO)/(9*SUM(B.IP)) AS SO9
    , SUM(B.SO)/SUM(B.PT) AS SOPercent
    , IF(SUM(B.IP) = 0
        , 0
        , 3 + ((13*SUM(B.HR)+3*(SUM(B.BB+B.HBP)-2*B.SO))/SUM(B.IP))
        ) AS DICE
    , IF(SUM(B.IP) = 0
        , 0
        , (SUM(B.H)+SUM(B.BB)) / SUM(B.IP)) AS WHIP
    , IF(SUM(B.IP)+SUM(B.BF) = 0
        , 0
        , 9*((SUM(B.H)+SUM(B.BB)+SUM(B.HBP))* SUM(B.PTB))/(SUM(B.BF)*SUM(B.IP))
              -0.56) AS CERA
FROM dummy_table A LEFT JOIN dummy_table B
    ON A.pitcher = B.pitcher
        AND A.game_date > B.game_date
        AND B.game_date >= DATE_SUB(A.game_date, INTERVAL 100 DAY)
GROUP BY A.pitcher, A.game_date
ORDER BY A.game_id, A.game_date, A.pitcher
;

ALTER TABLE rolling_start_pitcher ADD PRIMARY KEY (game_id, pitcher);


CREATE TEMPORARY TABLE start_pitcher ENGINE=MEMORY AS   -- noqa: PRS
SELECT
    hsp.game_id
    , hsp.game_date
    , hsp.pitcher
    , hsp.homeTeam
    , IF(hsp.BFP IS NULL, 0, hsp.BFP) AS BFP_HIST
    , IF(hsp.IP IS NULL, 0, hsp.IP) AS IP_HIST
    , IF(hsp.BB9 IS NULL, 0, hsp.BB9) AS BB9_HIST
    , IF(hsp.HA9 IS NULL, 0, hsp.HA9) AS HA9_HIST
    , IF(hsp.HRA9 IS NULL, 0, hsp.HRA9) AS HRA9_HIST
    , IF(hsp.SO9 IS NULL, 0, hsp.SO9) AS SO9_HIST
    , IF(hsp.SOPercent IS NULL, 0, hsp.SOPercent) AS SOPercent_HIST
    , IF(hsp.DICE IS NULL, 0, hsp.DICE) AS DICE_HIST
    , IF(hsp.WHIP IS NULL, 0, hsp.WHIP) AS WHIP_HIST
    , IF(hsp.CERA IS NULL, 0, hsp.CERA) AS CERA_HIST
    , IF(rsp.BFP IS NULL, 0, rsp.BFP) AS BFP_ROLL
    , IF(rsp.IP IS NULL, 0, rsp.IP) AS IP_ROLL
    , IF(rsp.BB9 IS NULL, 0, rsp.BB9) AS BB9_ROLL
    , IF(rsp.HA9 IS NULL, 0, rsp.HA9) AS HA9_ROLL
    , IF(rsp.HRA9 IS NULL, 0, rsp.HRA9) AS HRA9_ROLL
    , IF(rsp.SO9 IS NULL, 0, rsp.SO9) AS SO9_ROLL
    , IF(rsp.SOPercent IS NULL, 0, rsp.SOPercent) AS SOPercent_ROLL
    , IF(rsp.DICE IS NULL, 0, rsp.DICE) AS DICE_ROLL
    , IF(rsp.WHIP IS NULL, 0, rsp.WHIP) AS WHIP_ROLL
    , IF(rsp.CERA IS NULL, 0, rsp.CERA) AS CERA_ROLL
    , hsp.HTWins
FROM historic_start_pitcher hsp JOIN rolling_start_pitcher rsp
    ON hsp.game_id=rsp.game_id
           AND hsp.pitcher=rsp.pitcher
ORDER BY hsp.game_id
;

ALTER TABLE start_pitcher ADD PRIMARY KEY (game_id, pitcher);


CREATE TEMPORARY TABLE home_start_pitcher ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM start_pitcher WHERE homeTeam=1
;

ALTER TABLE home_start_pitcher ADD PRIMARY KEY (game_id, pitcher);


CREATE TEMPORARY TABLE away_start_pitcher ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM start_pitcher WHERE homeTeam=0
;

ALTER TABLE away_start_pitcher ADD PRIMARY KEY (game_id, pitcher);


CREATE OR REPLACE TABLE features AS
SELECT
    h.game_id
    , h.game_date
    , h.BFP_HIST - a.BFP_HIST AS SP_DIFF_HM_AY_BFP_HIST
    , h.IP_HIST - a.IP_HIST AS SP_DIFF_HM_AY_IP_HIST
    , h.BB9_HIST - a.BB9_HIST AS SP_DIFF_HM_AY_BB9_HIST
    , h.HA9_HIST - a.HA9_HIST AS SP_DIFF_HM_AY_HA9_HIST
    , h.HRA9_HIST - a.HRA9_HIST AS SP_DIFF_HM_AY_HRA9_HIST
    , h.SO9_HIST - a.SO9_HIST AS SP_DIFF_HM_AY_SO9_HIST
    , h.SOPercent_HIST - a.SOPercent_HIST AS SP_DIFF_HM_AY_SOPercent_HIST
    , h.DICE_HIST - a.DICE_HIST AS SP_DIFF_HM_AY_DICE_HIST
    , h.WHIP_HIST - a.WHIP_HIST AS SP_DIFF_HM_AY_WHIP_HIST
    , h.CERA_HIST - a.CERA_HIST AS SP_DIFF_HM_AY_CERA_HIST
    , h.BFP_ROLL - a.BFP_ROLL AS SP_DIFF_HM_AY_BFP_ROLL
    , h.IP_ROLL - a.IP_ROLL AS SP_DIFF_HM_AY_IP_ROLL
    , h.BB9_ROLL - a.BB9_ROLL AS SP_DIFF_HM_AY_BB9_ROLL
    , h.HA9_ROLL - a.HA9_ROLL AS SP_DIFF_HM_AY_HA9_ROLL
    , h.HRA9_ROLL - a.HRA9_ROLL AS SP_DIFF_HM_AY_HRA9_ROLL
    , h.SO9_ROLL - a.SO9_ROLL AS SP_DIFF_HM_AY_SO9_ROLL
    , h.SOPercent_ROLL - a.SOPercent_ROLL AS SP_DIFF_HM_AY_SOPercent_ROLL
    , h.DICE_ROLL - a.DICE_ROLL AS SP_DIFF_HM_AY_DICE_ROLL
    , h.WHIP_ROLL - a.WHIP_ROLL AS SP_DIFF_HM_AY_WHIP_ROLL
    , h.CERA_ROLL - a.CERA_ROLL AS SP_DIFF_HM_AY_CERA_ROLL
    , h.HTWins
FROM home_start_pitcher h JOIN away_start_pitcher a
    ON h.game_id = a.game_id
;

ALTER TABLE features ADD PRIMARY KEY (game_id, game_date);
