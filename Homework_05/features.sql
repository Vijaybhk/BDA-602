USE baseball;

## Starting Pitcher stats historic for home and away teams by game id
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
        , 3 + ((13*SUM(B.HR)+3*(SUM(B.BB+B.HBP)-2*B.SO))/SUM(B.IP))) AS DICE
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


## Starting Pitcher stats rolling 100day for home and away teams by game id
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
        , 3 + ((13*SUM(B.HR)+3*(SUM(B.BB+B.HBP)-2*B.SO))/SUM(B.IP))) AS DICE
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


## Starting Pitcher stats combined historic and rolling for home and away teams by game id
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


## Starting Pitcher stats split for home and away teams by game id
CREATE TEMPORARY TABLE home_start_pitcher ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM start_pitcher WHERE homeTeam=1
;

ALTER TABLE home_start_pitcher ADD PRIMARY KEY (game_id, pitcher);


CREATE TEMPORARY TABLE away_start_pitcher ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM start_pitcher WHERE homeTeam=0
;

ALTER TABLE away_start_pitcher ADD PRIMARY KEY (game_id, pitcher);
# -----------------------------------------------------------------------------


## Team batting historic stats for home and away teams by game id
CREATE TEMPORARY TABLE hist_team_batting ENGINE=MEMORY AS   -- noqa: PRS
WITH batting_dummy (game_id, game_date, team_id, HomeTeam, H, AB, HR, K, SF,
    BB, HBP, 1B, 2B, 3B, TB) AS (
    SELECT
        tbc.game_id
        , DATE(g.local_date)
        , tbc.team_id
        , tbc.homeTeam
        , tbc.Hit
        , tbc.atBat
        , tbc.Home_Run
        , tbc.Strikeout
        , tbc.Sac_Fly
        , tbc.Walk
        , tbc.Hit_By_Pitch
        , tbc.Single
        , tbc.Double
        , tbc.Triple
        , tbc.Single+(2*tbc.Double)+(3*tbc.Triple)+(4*tbc.Home_Run) AS TB
    FROM game g
        JOIN team_batting_counts tbc on g.game_id = tbc.game_id
    GROUP BY tbc.game_id, tbc.team_id
    ORDER BY tbc.game_id, tbc.team_id
)
SELECT
    A.game_id
    , A.game_date
    , A.team_id
    , A.HomeTeam
    , IF(SUM(B.AB)=0, 0, SUM(B.H)/SUM(B.AB)) AS AVG
    , IF(SUM(B.AB-B.K-B.HR+B.SF)=0
        , 0
        , SUM(B.H-B.HR)/SUM(B.AB-B.K-B.HR+B.SF)) AS BABIP
    , IF(SUM(B.AB+B.BB+B.HBP+B.SF)=0
        , 0
        , SUM(B.H+B.BB+B.HBP)/SUM(B.AB+B.BB+B.HBP+B.SF)) AS OBP
    , IF(SUM(B.AB)=0
        , 0
        , SUM(B.TB)/SUM(B.AB)) AS SLG
    , IF(SUM(B.AB)*SUM(B.AB+B.BB+B.HBP+B.SF)=0
        , 0
        , SUM(B.H+B.BB+B.HBP)/SUM(B.AB+B.BB+B.HBP+B.SF) + SUM(B.TB)/SUM(B.AB)
        ) AS OPS
FROM batting_dummy A LEFT JOIN batting_dummy B
    ON A.team_id = B.team_id
        AND A.game_date > B.game_date
GROUP BY A.game_id, A.team_id
ORDER BY A.game_id, A.team_id
;

ALTER TABLE hist_team_batting ADD PRIMARY KEY (game_id, team_id);


## Team batting rolling 100 day stats for home and away teams by game id
CREATE TEMPORARY TABLE roll_team_batting ENGINE=MEMORY AS   -- noqa: PRS
WITH batting_dummy (game_id, game_date, team_id, HomeTeam, H, AB, HR, K, SF,
    BB, HBP, 1B, 2B, 3B, TB) AS (
    SELECT
        tbc.game_id
        , DATE(g.local_date)
        , tbc.team_id
        , tbc.homeTeam
        , tbc.Hit
        , tbc.atBat
        , tbc.Home_Run
        , tbc.Strikeout
        , tbc.Sac_Fly
        , tbc.Walk
        , tbc.Hit_By_Pitch
        , tbc.Single
        , tbc.Double
        , tbc.Triple
        , tbc.Single+(2*tbc.Double)+(3*tbc.Triple)+(4*tbc.Home_Run) AS TB
    FROM game g
        JOIN team_batting_counts tbc on g.game_id = tbc.game_id
    GROUP BY tbc.game_id, tbc.team_id
    ORDER BY tbc.game_id, tbc.team_id
)
SELECT
    A.game_id
    , A.game_date
    , A.team_id
    , A.HomeTeam
    , IF(SUM(B.AB)=0, 0, SUM(B.H)/SUM(B.AB)) AS AVG
    , IF(SUM(B.AB-B.K-B.HR+B.SF)=0
        , 0
        , SUM(B.H-B.HR)/SUM(B.AB-B.K-B.HR+B.SF)) AS BABIP
    , IF(SUM(B.AB+B.BB+B.HBP+B.SF)=0
        , 0
        , SUM(B.H+B.BB+B.HBP)/SUM(B.AB+B.BB+B.HBP+B.SF)) AS OBP
    , IF(SUM(B.AB)=0
        , 0
        , SUM(B.TB)/SUM(B.AB)) AS SLG
    , IF(SUM(B.AB)*SUM(B.AB+B.BB+B.HBP+B.SF)=0
        , 0
        , SUM(B.H+B.BB+B.HBP)/SUM(B.AB+B.BB+B.HBP+B.SF) + SUM(B.TB)/SUM(B.AB)
        ) AS OPS
FROM batting_dummy A LEFT JOIN batting_dummy B
    ON A.team_id = B.team_id
        AND A.game_date > B.game_date
           AND B.game_date >= DATE_SUB(A.game_date, INTERVAL 100 DAY)
GROUP BY A.game_id, A.team_id
ORDER BY A.game_id, A.team_id
;

ALTER TABLE roll_team_batting ADD PRIMARY KEY (game_id, team_id);


## Team batting combined historic and rolling stats for home and away teams by game id
CREATE TEMPORARY TABLE team_batting ENGINE=MEMORY AS   -- noqa: PRS
SELECT
    htb.game_id
    , htb.game_date
    , htb.team_id
    , htb.homeTeam
    , IF(htb.AVG IS NULL, 0, htb.AVG) AS AVG_HIST
    , IF(htb.BABIP IS NULL, 0, htb.BABIP) AS BABIP_HIST
    , IF(htb.OBP IS NULL, 0, htb.OBP) AS OBP_HIST
    , IF(htb.SLG IS NULL, 0, htb.SLG) AS SLG_HIST
    , IF(htb.OPS IS NULL, 0, htb.OPS) AS OPS_HIST
    , IF(rtb.AVG IS NULL, 0, rtb.AVG) AS AVG_ROLL
    , IF(htb.BABIP IS NULL, 0, htb.BABIP) AS BABIP_ROLL
    , IF(rtb.OBP IS NULL, 0, rtb.OBP) AS OBP_ROLL
    , IF(rtb.SLG IS NULL, 0, rtb.SLG) AS SLG_ROLL
    , IF(rtb.OPS IS NULL, 0, rtb.OPS) AS OPS_ROLL
FROM hist_team_batting htb JOIN roll_team_batting rtb
    ON htb.game_id=rtb.game_id
           AND htb.team_id=rtb.team_id
ORDER BY htb.game_id
;

ALTER TABLE team_batting ADD PRIMARY KEY (game_id, team_id);


## Team batting stats split for home and away teams by game id
CREATE TEMPORARY TABLE home_team_batter ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM team_batting WHERE homeTeam=1
;

ALTER TABLE home_team_batter ADD PRIMARY KEY (game_id, team_id);


CREATE TEMPORARY TABLE away_team_batter ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM team_batting WHERE homeTeam=0
;

ALTER TABLE away_team_batter ADD PRIMARY KEY (game_id, team_id);
# -----------------------------------------------------------------------------


## Team pitching historic stats for home and away teams by game id
CREATE TEMPORARY TABLE hist_team_pitching ENGINE=MEMORY AS   -- noqa: PRS
WITH pitcher_dummy (game_id, game_date, team_id, HomeTeam, H, AB, HR, K, SF,
    BB, HBP, IP, BF, IBB, PT, PTB) AS (
    SELECT
        pc.game_id
        , DATE(g.local_date)
        , pc.team_id
        , pc.homeTeam
        , SUM(pc.Hit)
        , SUM(pc.atBat)
        , SUM(pc.Home_Run)
        , SUM(pc.Strikeout)
        , SUM(pc.Sac_Fly)
        , SUM(pc.Walk)
        , SUM(pc.Hit_By_Pitch)
        , SUM(pc.outsPlayed)/3
        , SUM(pc.plateApperance)
        , SUM(pc.Intent_Walk)
        , SUM(pc.pitchesThrown)
        , SUM(0.89*(1.255*(pc.Hit-pc.Home_Run) + 4*pc.Home_Run)
              + 0.56*(pc.Walk+pc.Hit_By_Pitch-pc.Intent_Walk)) AS PTB
    FROM game g
        JOIN pitcher_counts pc on g.game_id = pc.game_id
    GROUP BY pc.game_id, pc.team_id
    ORDER BY pc.game_id, pc.team_id
)
SELECT
    A.game_id
    , A.game_date
    , A.team_id
    , A.HomeTeam
    , SUM(B.BF) AS BFP
    , SUM(B.IP) AS IP
    , SUM(B.BB)/9 AS BB9
    , SUM(B.H)/9 AS HA9
    , SUM(B.HR)/9 AS HRA9
    , SUM(B.K)/(9*SUM(B.IP)) AS SO9
    , SUM(B.K)/SUM(B.PT) AS SOPP
    , IF(SUM(B.IP) = 0
        , 0
        , 3 + ((13*SUM(B.HR)+3*(SUM(B.BB+B.HBP)-2*B.K))/SUM(B.IP))) AS DICE
    , IF(SUM(B.IP) = 0
        , 0
        , (SUM(B.H)+SUM(B.BB)) / SUM(B.IP)) AS WHIP
    , IF(SUM(B.IP)+SUM(B.BF) = 0
        , 0
        , 9*((SUM(B.H)+SUM(B.BB)+SUM(B.HBP))* SUM(B.PTB))/(SUM(B.BF)*SUM(B.IP))
              -0.56) AS CERA
FROM pitcher_dummy A LEFT JOIN pitcher_dummy B
    ON A.team_id = B.team_id
        AND A.game_date > B.game_date
GROUP BY A.game_id, A.team_id
ORDER BY A.game_id, A.team_id
;

ALTER TABLE hist_team_pitching ADD PRIMARY KEY (game_id, team_id);


## Team pitching rolling 100 day stats for home and away teams by game id
CREATE TEMPORARY TABLE roll_team_pitching ENGINE=MEMORY AS   -- noqa: PRS
WITH pitcher_dummy (game_id, game_date, team_id, HomeTeam, H, AB, HR, K, SF,
    BB, HBP, IP, BF, IBB, PT, PTB) AS (
    SELECT
        pc.game_id
        , DATE(g.local_date)
        , pc.team_id
        , pc.homeTeam
        , SUM(pc.Hit)
        , SUM(pc.atBat)
        , SUM(pc.Home_Run)
        , SUM(pc.Strikeout)
        , SUM(pc.Sac_Fly)
        , SUM(pc.Walk)
        , SUM(pc.Hit_By_Pitch)
        , SUM(pc.outsPlayed)/3
        , SUM(pc.plateApperance)
        , SUM(pc.Intent_Walk)
        , SUM(pc.pitchesThrown)
        , SUM(0.89*(1.255*(pc.Hit-pc.Home_Run) + 4*pc.Home_Run)
              + 0.56*(pc.Walk+pc.Hit_By_Pitch-pc.Intent_Walk)) AS PTB
    FROM game g
        JOIN pitcher_counts pc on g.game_id = pc.game_id
    GROUP BY pc.game_id, pc.team_id
    ORDER BY pc.game_id, pc.team_id
)
SELECT
    A.game_id
    , A.game_date
    , A.team_id
    , A.HomeTeam
    , SUM(B.BF) AS BFP
    , SUM(B.IP) AS IP
    , SUM(B.BB)/9 AS BB9
    , SUM(B.H)/9 AS HA9
    , SUM(B.HR)/9 AS HRA9
    , SUM(B.K)/(9*SUM(B.IP)) AS SO9
    , SUM(B.K)/SUM(B.PT) AS SOPP
    , IF(SUM(B.IP) = 0
        , 0
        , 3 + ((13*SUM(B.HR)+3*(SUM(B.BB+B.HBP)-2*B.K))/SUM(B.IP))) AS DICE
    , IF(SUM(B.IP) = 0
        , 0
        , (SUM(B.H)+SUM(B.BB)) / SUM(B.IP)) AS WHIP
    , IF(SUM(B.IP)+SUM(B.BF) = 0
        , 0
        , 9*((SUM(B.H)+SUM(B.BB)+SUM(B.HBP))* SUM(B.PTB))/(SUM(B.BF)*SUM(B.IP))
              -0.56) AS CERA
FROM pitcher_dummy A LEFT JOIN pitcher_dummy B
    ON A.team_id = B.team_id
        AND A.game_date > B.game_date
           AND B.game_date >= DATE_SUB(A.game_date, INTERVAL 100 DAY)
GROUP BY A.game_id, A.team_id
ORDER BY A.game_id, A.team_id
;

ALTER TABLE roll_team_pitching ADD PRIMARY KEY (game_id, team_id);


## Team pitching combined historic and rolling stats for home and away teams by game id
CREATE TEMPORARY TABLE team_pitching ENGINE=MEMORY AS   -- noqa: PRS
SELECT
    htp.game_id
    , htp.game_date
    , htp.team_id
    , htp.homeTeam
    , IF(htp.BFP IS NULL, 0, htp.BFP) AS BFP_HIST
    , IF(htp.IP IS NULL, 0, htp.IP) AS IP_HIST
    , IF(htp.BB9 IS NULL, 0, htp.BB9) AS BB9_HIST
    , IF(htp.HA9 IS NULL, 0, htp.HA9) AS HA9_HIST
    , IF(htp.HRA9 IS NULL, 0, htp.HRA9) AS HRA9_HIST
    , IF(htp.SO9 IS NULL, 0, htp.SO9) AS SO9_HIST
    , IF(htp.SOPP IS NULL, 0, htp.SOPP) AS SOPercent_HIST
    , IF(htp.DICE IS NULL, 0, htp.DICE) AS DICE_HIST
    , IF(htp.WHIP IS NULL, 0, htp.WHIP) AS WHIP_HIST
    , IF(htp.CERA IS NULL, 0, htp.CERA) AS CERA_HIST
    , IF(rtp.BFP IS NULL, 0, rtp.BFP) AS BFP_ROLL
    , IF(rtp.IP IS NULL, 0, rtp.IP) AS IP_ROLL
    , IF(rtp.BB9 IS NULL, 0, rtp.BB9) AS BB9_ROLL
    , IF(rtp.HA9 IS NULL, 0, rtp.HA9) AS HA9_ROLL
    , IF(rtp.HRA9 IS NULL, 0, rtp.HRA9) AS HRA9_ROLL
    , IF(rtp.SO9 IS NULL, 0, rtp.SO9) AS SO9_ROLL
    , IF(rtp.SOPP IS NULL, 0, rtp.SOPP) AS SOPP_ROLL
    , IF(rtp.DICE IS NULL, 0, rtp.DICE) AS DICE_ROLL
    , IF(rtp.WHIP IS NULL, 0, rtp.WHIP) AS WHIP_ROLL
    , IF(rtp.CERA IS NULL, 0, rtp.CERA) AS CERA_ROLL
FROM hist_team_pitching htp JOIN roll_team_pitching rtp
    ON htp.game_id=rtp.game_id
           AND htp.team_id=rtp.team_id
ORDER BY htp.game_id
;

ALTER TABLE team_pitching ADD PRIMARY KEY (game_id, team_id);


## Team pitching stats split for home and away teams by game id
CREATE TEMPORARY TABLE home_team_pitcher ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM team_pitching WHERE homeTeam=1
;

ALTER TABLE home_team_pitcher ADD PRIMARY KEY (game_id, team_id);


CREATE TEMPORARY TABLE away_team_pitcher ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM team_pitching WHERE homeTeam=0
;

ALTER TABLE away_team_pitcher ADD PRIMARY KEY (game_id, team_id);
# -----------------------------------------------------------------------------


## Final features/stats differences between home and away team stats by game id
## Split features into three final tables to make code execute fast
CREATE OR REPLACE TABLE start_pitcher_features AS
SELECT
    hsp.game_id
    , hsp.game_date
    , hsp.BFP_HIST - asp.BFP_HIST AS SP_BFP_DIFF_HIST
    , hsp.IP_HIST - asp.IP_HIST AS SP_IP_DIFF_HIST
    , hsp.BB9_HIST - asp.BB9_HIST AS SP_BB9_DIFF_HIST
    , hsp.HA9_HIST - asp.HA9_HIST AS SP_HA9_DIFF_HIST
    , hsp.HRA9_HIST - asp.HRA9_HIST AS SP_HRA9_DIFF_HIST
    , hsp.SO9_HIST - asp.SO9_HIST AS SP_SO9_DIFF_HIST
    , hsp.SOPercent_HIST - asp.SOPercent_HIST AS SP_SOPP_DIFF_HIST
    , hsp.DICE_HIST - asp.DICE_HIST AS SP_DICE_DIFF_HIST
    , hsp.WHIP_HIST - asp.WHIP_HIST AS SP_WHIP_DIFF_HIST
    , hsp.CERA_HIST - asp.CERA_HIST AS SP_CERA_DIFF_HIST
    , hsp.BFP_ROLL - asp.BFP_ROLL AS SP_BFP_DIFF_ROLL
    , hsp.IP_ROLL - asp.IP_ROLL AS SP_IP_DIFF_ROLL
    , hsp.BB9_ROLL - asp.BB9_ROLL AS SP_BB9_DIFF_ROLL
    , hsp.HA9_ROLL - asp.HA9_ROLL AS SP_HA9_DIFF_ROLL
    , hsp.HRA9_ROLL - asp.HRA9_ROLL AS SP_HRA9_DIFF_ROLL
    , hsp.SO9_ROLL - asp.SO9_ROLL AS SP_SO9_DIFF_ROLL
    , hsp.SOPercent_ROLL - asp.SOPercent_ROLL AS SP_SOPP_DIFF_ROLL
    , hsp.DICE_ROLL - asp.DICE_ROLL AS SP_DICE_DIFF_ROLL
    , hsp.WHIP_ROLL - asp.WHIP_ROLL AS SP_WHIP_DIFF_ROLL
    , hsp.CERA_ROLL - asp.CERA_ROLL AS SP_CERA_DIFF_ROLL
    , hsp.HTWins
FROM home_start_pitcher hsp JOIN away_start_pitcher asp
    ON hsp.game_id = asp.game_id
;

ALTER TABLE start_pitcher_features ADD PRIMARY KEY (game_id, game_date);


CREATE OR REPLACE TABLE team_batter_features AS
SELECT
    htb.game_id
    , htb.game_date
    , htb.AVG_HIST - atb.AVG_HIST AS TB_AVG_DIFF_HIST
    , htb.BABIP_HIST - atb.BABIP_HIST AS TB_BABIP_DIFF_HIST
    , htb.OBP_HIST - atb.OBP_HIST AS TB_OBP_DIFF_HIST
    , htb.SLG_HIST - atb.SLG_HIST AS TB_SLG_DIFF_HIST
    , htb.OPS_HIST - atb.SLG_HIST AS TB_OPS_DIFF_HIST
    , htb.AVG_ROLL - atb.AVG_ROLL AS TB_AVG_DIFF_ROLL
    , htb.BABIP_ROLL - atb.BABIP_ROLL AS TB_BABIP_DIFF_ROLL
    , htb.OBP_ROLL - atb.OBP_ROLL AS TB_OBP_DIFF_ROLL
    , htb.SLG_ROLL - atb.SLG_ROLL AS TB_SLG_DIFF_ROLL
    , htb.OPS_ROLL - atb.OPS_ROLL AS TB_OPS_DIFF_ROLL
FROM home_team_batter htb JOIN away_team_batter atb
    ON htb.game_id = atb.game_id
;

ALTER TABLE team_batter_features ADD PRIMARY KEY (game_id, game_date);


CREATE OR REPLACE TABLE team_pitcher_features AS
SELECT
    htp.game_id
    , htp.game_date
    , htp.BFP_HIST - atp.BFP_HIST AS TP_BFP_DIFF_HIST
    , htp.IP_HIST - atp.IP_HIST AS TP_IP_DIFF_HIST
    , htp.BB9_HIST - atp.BB9_HIST AS TP_BB9_DIFF_HIST
    , htp.HA9_HIST - atp.HA9_HIST AS TP_HA9_DIFF_HIST
    , htp.HRA9_HIST - atp.HRA9_HIST AS TP_HRA9_DIFF_HIST
    , htp.SO9_HIST - atp.SO9_HIST AS TP_SO9_DIFF_HIST
    , htp.SOPercent_HIST - atp.SOPercent_HIST AS TP_SOPP_DIFF_HIST
    , htp.DICE_HIST - atp.DICE_HIST AS TP_DICE_DIFF_HIST
    , htp.WHIP_HIST - atp.WHIP_HIST AS TP_WHIP_DIFF_HIST
    , htp.CERA_HIST - atp.CERA_HIST AS TP_CERA_DIFF_HIST
    , htp.BFP_ROLL - atp.BFP_ROLL AS TP_BFP_DIFF_ROLL
    , htp.IP_ROLL - atp.IP_ROLL AS TP_IP_DIFF_ROLL
    , htp.BB9_ROLL - atp.BB9_ROLL AS TP_BB9_DIFF_ROLL
    , htp.HA9_ROLL - atp.HA9_ROLL AS TP_HA9_DIFF_ROLL
    , htp.HRA9_ROLL - atp.HRA9_ROLL AS TP_HRA9_DIFF_ROLL
    , htp.SO9_ROLL - atp.SO9_ROLL AS TP_SO9_DIFF_ROLL
    , htp.SOPP_ROLL - atp.SOPP_ROLL AS TP_SOPP_DIFF_ROLL
    , htp.DICE_ROLL - atp.DICE_ROLL AS TP_DICE_DIFF_ROLL
    , htp.WHIP_ROLL - atp.WHIP_ROLL AS TP_WHIP_DIFF_ROLL
    , htp.CERA_ROLL - atp.CERA_ROLL AS TP_CERA_DIFF_ROLL
FROM home_team_pitcher htp JOIN away_team_pitcher atp
    ON htp.game_id = atp.game_id
;

ALTER TABLE team_pitcher_features ADD PRIMARY KEY (game_id, game_date);

SELECT * from team_batter_features;
