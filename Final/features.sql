USE baseball;

## Starting Pitcher stats historic for home and away teams by game id
CREATE TEMPORARY TABLE hist_sp_stats ENGINE=MEMORY AS  -- noqa: PRS
WITH dummy_table (game_id, game_date, pitcher, startingPitcher, IP, homeTeam,
    awayTeam, HR, H, BB, AB, BF, IBB, HBP, K, PT, PTB, HTRuns, ATRuns, HTWins)
    AS (
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
        , pc.atBat
        , pc.plateApperance
        , pc.Intent_Walk
        , pc.Hit_By_Pitch
        , pc.Strikeout
        , pc.pitchesThrown
        , 0.89*(1.255*(pc.Hit-pc.Home_Run) + 4*pc.Home_Run)
              + 0.56*(pc.Walk+pc.Hit_By_Pitch-pc.Intent_Walk) AS PTB
        , bs.home_runs
        , bs.away_runs
        , IF(bs.home_runs>bs.away_runs, 1, 0) AS HTWins
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
    , A.pitcher AS start_pitcher
    , A.homeTeam
    , SUM(B.BF) AS BFP_HIST
    , SUM(B.IP) AS IP_HIST
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.BB)/SUM(B.IP)))  AS BB9_HIST
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.H)/SUM(B.IP))) AS HA9_HIST
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.HR)/SUM(B.IP))) AS HRA9_HIST
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.K)/SUM(B.IP))) AS SO9_HIST
    , IF(SUM(B.PT)=0, NULL, SUM(B.K)/SUM(B.PT)) AS SOPP_HIST
    , IF(SUM(B.IP) = 0
        , NULL
        , (13*SUM(B.HR)+3*SUM(B.BB)-2*SUM(B.K))/SUM(B.IP)) AS FIP_HIST
    , IF(SUM(B.IP) = 0
        , NULL
        , (SUM(B.H)+SUM(B.BB)) / SUM(B.IP)) AS WHIP_HIST
    , IF(SUM(B.IP)*SUM(B.BF) = 0
        , NULL
        , 9*((SUM(B.H)+SUM(B.BB)+SUM(B.HBP))* SUM(B.PTB))/(SUM(B.BF)*SUM(B.IP))
              -0.56) AS CERA_HIST
    , IF(SUM(B.BB)=0, NULL, SUM(B.K)/SUM(B.BB)) AS SWR_HIST
    , IF(SUM(B.AB)=0, NULL, SUM(B.H)/SUM(B.AB)) AS OBA_HIST
    , IF(SUM(B.IP)=0, NULL, (SUM(B.K)+SUM(B.BB))/SUM(B.IP)) AS PFR_HIST
    , IF(SUM(B.IP)=0
        , NULL
        , 9*(((SUM(B.H)+SUM(B.BB))*SUM(B.PTB))/(SUM(B.AB)+SUM(B.BB)))/SUM(B.IP)
        ) AS RAA_HIST
    , A.HTWins
FROM dummy_table A LEFT JOIN dummy_table B
    ON A.pitcher = B.pitcher
        AND A.game_date > B.game_date
GROUP BY A.pitcher, A.game_date
ORDER BY A.game_id, A.game_date, A.pitcher
;

ALTER TABLE hist_sp_stats ADD PRIMARY KEY (game_id, start_pitcher);


## Starting Pitcher stats rolling 100day for home and away teams by game id
CREATE TEMPORARY TABLE roll_sp_stats ENGINE=MEMORY AS   -- noqa: PRS
WITH dummy_table (game_id, game_date, pitcher, startingPitcher, IP,
    homeTeam, awayTeam, HR, H, BB, AB, BF, IBB, HBP, K, PT, PTB) AS (
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
        , pc.atBat
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
    , A.pitcher AS start_pitcher
    , A.homeTeam
    , SUM(B.BF) AS BFP_ROLL
    , SUM(B.IP) AS IP_ROLL
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.BB)/SUM(B.IP)))  AS BB9_ROLL
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.H)/SUM(B.IP))) AS HA9_ROLL
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.HR)/SUM(B.IP))) AS HRA9_ROLL
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.K)/SUM(B.IP))) AS SO9_ROLL
    , IF(SUM(B.PT)=0, NULL, SUM(B.K)/SUM(B.PT)) AS SOPP_ROLL
    , IF(SUM(B.IP) = 0
        , NULL
        , (13*SUM(B.HR)+3*SUM(B.BB)-2*SUM(B.K))/SUM(B.IP)) AS FIP_ROLL
    , IF(SUM(B.IP) = 0
        , NULL
        , (SUM(B.H)+SUM(B.BB)) / SUM(B.IP)) AS WHIP_ROLL
    , IF(SUM(B.IP)*SUM(B.BF) = 0
        , NULL
        , 9*((SUM(B.H)+SUM(B.BB)+SUM(B.HBP))* SUM(B.PTB))/(SUM(B.BF)*SUM(B.IP))
              -0.56) AS CERA_ROLL
    , IF(SUM(B.BB)=0, NULL, SUM(B.K)/SUM(B.BB)) AS SWR_ROLL
    , IF(SUM(B.AB)=0, NULL, SUM(B.H)/SUM(B.AB)) AS OBA_ROLL
    , IF(SUM(B.IP)=0, NULL, (SUM(B.K)+SUM(B.BB))/SUM(B.IP)) AS PFR_ROLL
    , IF(SUM(B.IP)=0
        , NULL
        , 9*(((SUM(B.H)+SUM(B.BB))*SUM(B.PTB))/(SUM(B.AB)+SUM(B.BB)))/SUM(B.IP)
        ) AS RAA_ROLL
FROM dummy_table A LEFT JOIN dummy_table B
    ON A.pitcher = B.pitcher
        AND A.game_date > B.game_date
        AND B.game_date >= DATE_SUB(A.game_date, INTERVAL 100 DAY)
GROUP BY A.pitcher, A.game_date
ORDER BY A.game_id, A.game_date, A.pitcher
;

ALTER TABLE roll_sp_stats ADD PRIMARY KEY (game_id, start_pitcher);


## Starting Pitcher stats combined historic and rolling for home and away teams by game id
CREATE  TEMPORARY TABLE sp_stats ENGINE=MEMORY AS   -- noqa: PRS
SELECT
    hsp.*
    , rsp.BFP_ROLL
    , rsp.IP_ROLL
    , rsp.BB9_ROLL
    , rsp.HA9_ROLL
    , rsp.HRA9_ROLL
    , rsp.SO9_ROLL
    , rsp.SOPP_ROLL
    , rsp.FIP_ROLL
    , rsp.CERA_ROLL
    , rsp.WHIP_ROLL
    , rsp.SWR_ROLL
    , rsp.PFR_ROLL
    , rsp.OBA_ROLL
    , rsp.RAA_ROLL
FROM hist_sp_stats hsp JOIN roll_sp_stats rsp
    ON hsp.game_id=rsp.game_id
           AND hsp.start_pitcher=rsp.start_pitcher
ORDER BY hsp.game_id
;

ALTER TABLE sp_stats ADD PRIMARY KEY (game_id, start_pitcher);

## Starting Pitcher stats split for home and away teams by game id
CREATE TEMPORARY TABLE home_sp_stats ENGINE=MEMORY AS  -- noqa: PRS
SELECT *
FROM sp_stats WHERE homeTeam=1
;

ALTER TABLE home_sp_stats ADD PRIMARY KEY (game_id, start_pitcher);


CREATE TEMPORARY TABLE away_sp_stats ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM sp_stats WHERE homeTeam=0
;

ALTER TABLE away_sp_stats ADD PRIMARY KEY (game_id, start_pitcher);

# Final starting pitcher features/stats differences between home and away team stats by game id
CREATE OR REPLACE TABLE sp_features AS
SELECT
    hsp.game_id
    , hsp.game_date
    , hsp.BFP_HIST - asp.BFP_HIST AS SP_BFP_DIFF_HIST
    , hsp.IP_HIST - asp.IP_HIST AS SP_IP_DIFF_HIST
    , hsp.BB9_HIST - asp.BB9_HIST AS SP_BB9_DIFF_HIST
    , hsp.HA9_HIST - asp.HA9_HIST AS SP_HA9_DIFF_HIST
    , hsp.HRA9_HIST - asp.HRA9_HIST AS SP_HRA9_DIFF_HIST
    , hsp.SO9_HIST - asp.SO9_HIST AS SP_SO9_DIFF_HIST
    , hsp.SOPP_HIST - asp.SOPP_HIST AS SP_SOPP_DIFF_HIST
    , hsp.FIP_HIST - asp.FIP_HIST AS SP_FIP_DIFF_HIST
    , hsp.WHIP_HIST - asp.WHIP_HIST AS SP_WHIP_DIFF_HIST
    , hsp.CERA_HIST - asp.CERA_HIST AS SP_CERA_DIFF_HIST
    , hsp.OBA_HIST - asp.OBA_HIST AS SP_OBA_DIFF_HIST
    , hsp.PFR_HIST - asp.PFR_HIST AS SP_PFR_DIFF_HIST
    , hsp.SWR_HIST - asp.SWR_HIST AS SP_SWR_DIFF_HIST
    , hsp.RAA_HIST - asp.RAA_HIST AS SP_RAA_DIFF_HIST
    , hsp.BFP_ROLL - asp.BFP_ROLL AS SP_BFP_DIFF_ROLL
    , hsp.IP_ROLL - asp.IP_ROLL AS SP_IP_DIFF_ROLL
    , hsp.BB9_ROLL - asp.BB9_ROLL AS SP_BB9_DIFF_ROLL
    , hsp.HA9_ROLL - asp.HA9_ROLL AS SP_HA9_DIFF_ROLL
    , hsp.HRA9_ROLL - asp.HRA9_ROLL AS SP_HRA9_DIFF_ROLL
    , hsp.SO9_ROLL - asp.SO9_ROLL AS SP_SO9_DIFF_ROLL
    , hsp.SOPP_ROLL - asp.SOPP_ROLL AS SP_SOPP_DIFF_ROLL
    , hsp.FIP_ROLL - asp.FIP_ROLL AS SP_FIP_DIFF_ROLL
    , hsp.WHIP_ROLL - asp.WHIP_ROLL AS SP_WHIP_DIFF_ROLL
    , hsp.CERA_ROLL - asp.CERA_ROLL AS SP_CERA_DIFF_ROLL
    , hsp.OBA_ROLL - asp.OBA_ROLL AS SP_OBA_DIFF_ROLL
    , hsp.PFR_ROLL - asp.PFR_ROLL AS SP_PFR_DIFF_ROLL
    , hsp.SWR_ROLL - asp.SWR_ROLL AS SP_SWR_DIFF_ROLL
    , hsp.RAA_ROLL - asp.RAA_ROLL AS SP_RAA_DIFF_ROLL
    , hsp.HTWins
FROM home_sp_stats hsp JOIN away_sp_stats asp
    ON hsp.game_id = asp.game_id
;

ALTER TABLE sp_features ADD PRIMARY KEY (game_id, game_date);

## ---------------------------------------------------------------------------


## Team pitching historic stats for home and away teams by game id
CREATE TEMPORARY TABLE hist_tp_stats ENGINE=MEMORY AS   -- noqa: PRS
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
    , SUM(B.BF) AS BFP_HIST
    , SUM(B.IP) AS IP_HIST
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.BB)/SUM(B.IP)))  AS BB9_HIST
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.H)/SUM(B.IP))) AS HA9_HIST
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.HR)/SUM(B.IP))) AS HRA9_HIST
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.K)/SUM(B.IP))) AS SO9_HIST
    , IF(SUM(B.PT)=0, NULL, SUM(B.K)/SUM(B.PT)) AS SOPP_HIST
    , IF(SUM(B.IP) = 0
        , NULL
        , (13*SUM(B.HR)+3*SUM(B.BB)-2*SUM(B.K))/SUM(B.IP)) AS FIP_HIST
    , IF(SUM(B.IP) = 0
        , NULL
        , (SUM(B.H)+SUM(B.BB)) / SUM(B.IP)) AS WHIP_HIST
    , IF(SUM(B.IP)*SUM(B.BF) = 0
        , NULL
        , 9*((SUM(B.H)+SUM(B.BB)+SUM(B.HBP))* SUM(B.PTB))/(SUM(B.BF)*SUM(B.IP))
              -0.56) AS CERA_HIST
    , IF(SUM(B.BB)=0, NULL, SUM(B.K)/SUM(B.BB)) AS SWR_HIST
    , IF(SUM(B.AB)=0, NULL, SUM(B.H)/SUM(B.AB)) AS OBA_HIST
    , IF(SUM(B.IP)=0, NULL, (SUM(B.K)+SUM(B.BB))/SUM(B.IP)) AS PFR_HIST
FROM pitcher_dummy A LEFT JOIN pitcher_dummy B
    ON A.team_id = B.team_id
        AND A.game_date > B.game_date
GROUP BY A.game_id, A.team_id
ORDER BY A.game_id, A.team_id
;

ALTER TABLE hist_tp_stats ADD PRIMARY KEY (game_id, team_id);


## Team pitching rolling 100 day stats for home and away teams by game id
CREATE TEMPORARY TABLE roll_tp_stats ENGINE=MEMORY AS   -- noqa: PRS
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
    , SUM(B.BF) AS BFP_ROLL
    , SUM(B.IP) AS IP_ROLL
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.BB)/SUM(B.IP)))  AS BB9_ROLL
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.H)/SUM(B.IP))) AS HA9_ROLL
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.HR)/SUM(B.IP))) AS HRA9_ROLL
    , IF(SUM(B.IP)=0, NULL, 9*(SUM(B.K)/SUM(B.IP))) AS SO9_ROLL
    , IF(SUM(B.PT)=0, NULL, SUM(B.K)/SUM(B.PT)) AS SOPP_ROLL
    , IF(SUM(B.IP) = 0
        , NULL
        , (13*SUM(B.HR)+3*SUM(B.BB)-2*SUM(B.K))/SUM(B.IP)) AS FIP_ROLL
    , IF(SUM(B.IP) = 0
        , NULL
        , (SUM(B.H)+SUM(B.BB)) / SUM(B.IP)) AS WHIP_ROLL
    , IF(SUM(B.IP)*SUM(B.BF) = 0
        , NULL
        , 9*((SUM(B.H)+SUM(B.BB)+SUM(B.HBP))* SUM(B.PTB))/(SUM(B.BF)*SUM(B.IP))
              -0.56) AS CERA_ROLL
    , IF(SUM(B.BB)=0, NULL, SUM(B.K)/SUM(B.BB)) AS SWR_ROLL
    , IF(SUM(B.AB)=0, NULL, SUM(B.H)/SUM(B.AB)) AS OBA_ROLL
    , IF(SUM(B.IP)=0, NULL, (SUM(B.K)+SUM(B.BB))/SUM(B.IP)) AS PFR_ROLL
FROM pitcher_dummy A LEFT JOIN pitcher_dummy B
    ON A.team_id = B.team_id
        AND A.game_date > B.game_date
           AND B.game_date >= DATE_SUB(A.game_date, INTERVAL 100 DAY)
GROUP BY A.game_id, A.team_id
ORDER BY A.game_id, A.team_id
;

ALTER TABLE roll_tp_stats ADD PRIMARY KEY (game_id, team_id);


## Team pitching combined historic and rolling stats for home and away teams by game id
CREATE TEMPORARY TABLE tp_stats ENGINE=MEMORY AS   -- noqa: PRS
SELECT
    htp.*
    , BFP_ROLL
    , IP_ROLL
    , BB9_ROLL
    , HA9_ROLL
    , HRA9_ROLL
    , SO9_ROLL
    , SOPP_ROLL
    , FIP_ROLL
    , WHIP_ROLL
    , CERA_ROLL
    , OBA_ROLL
    , PFR_ROLL
    , SWR_ROLL
FROM hist_tp_stats htp JOIN roll_tp_stats rtp
    ON htp.game_id=rtp.game_id
           AND htp.team_id=rtp.team_id
ORDER BY htp.game_id
;

ALTER TABLE tp_stats ADD PRIMARY KEY (game_id, team_id);


## Team pitching stats split for home and away teams by game id
CREATE TEMPORARY TABLE home_tp_stats ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM tp_stats WHERE homeTeam=1
;

ALTER TABLE home_tp_stats ADD PRIMARY KEY (game_id, team_id);


CREATE TEMPORARY TABLE away_tp_stats ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM tp_stats WHERE homeTeam=0
;

ALTER TABLE away_tp_stats ADD PRIMARY KEY (game_id, team_id);

CREATE OR REPLACE TABLE tp_features AS
SELECT
    htp.game_id
    , htp.game_date
    , htp.BFP_HIST - atp.BFP_HIST AS TP_BFP_DIFF_HIST
    , htp.IP_HIST - atp.IP_HIST AS TP_IP_DIFF_HIST
    , htp.BB9_HIST - atp.BB9_HIST AS TP_BB9_DIFF_HIST
    , htp.HA9_HIST - atp.HA9_HIST AS TP_HA9_DIFF_HIST
    , htp.HRA9_HIST - atp.HRA9_HIST AS TP_HRA9_DIFF_HIST
    , htp.SO9_HIST - atp.SO9_HIST AS TP_SO9_DIFF_HIST
    , htp.SOPP_HIST - atp.SOPP_HIST AS TP_SOPP_DIFF_HIST
    , htp.FIP_HIST - atp.FIP_HIST AS TP_FIP_DIFF_HIST
    , htp.WHIP_HIST - atp.WHIP_HIST AS TP_WHIP_DIFF_HIST
    , htp.CERA_HIST - atp.CERA_HIST AS TP_CERA_DIFF_HIST
    , htp.OBA_HIST - atp.OBA_HIST AS TP_OBA_DIFF_HIST
    , htp.PFR_HIST - atp.PFR_HIST AS TP_PFR_DIFF_HIST
    , htp.SWR_HIST - atp.SWR_HIST AS TP_SWR_DIFF_HIST
    , htp.BFP_ROLL - atp.BFP_ROLL AS TP_BFP_DIFF_ROLL
    , htp.IP_ROLL - atp.IP_ROLL AS TP_IP_DIFF_ROLL
    , htp.BB9_ROLL - atp.BB9_ROLL AS TP_BB9_DIFF_ROLL
    , htp.HA9_ROLL - atp.HA9_ROLL AS TP_HA9_DIFF_ROLL
    , htp.HRA9_ROLL - atp.HRA9_ROLL AS TP_HRA9_DIFF_ROLL
    , htp.SO9_ROLL - atp.SO9_ROLL AS TP_SO9_DIFF_ROLL
    , htp.SOPP_ROLL - atp.SOPP_ROLL AS TP_SOPP_DIFF_ROLL
    , htp.FIP_ROLL - atp.FIP_ROLL AS TP_FIP_DIFF_ROLL
    , htp.WHIP_ROLL - atp.WHIP_ROLL AS TP_WHIP_DIFF_ROLL
    , htp.CERA_ROLL - atp.CERA_ROLL AS TP_CERA_DIFF_ROLL
    , htp.OBA_ROLL - atp.OBA_ROLL AS TP_OBA_DIFF_ROLL
    , htp.PFR_ROLL - atp.PFR_ROLL AS TP_PFR_DIFF_ROLL
    , htp.SWR_ROLL - atp.SWR_ROLL AS TP_SWR_DIFF_ROLL
FROM home_tp_stats htp JOIN away_tp_stats atp
    ON htp.game_id = atp.game_id
;

ALTER TABLE tp_features ADD PRIMARY KEY (game_id, game_date);

## -------------------------------------------------------------------------

## Team batting historic stats for home and away teams by game id
CREATE TEMPORARY TABLE hist_tb_stats ENGINE=MEMORY AS   -- noqa: PRS
WITH batting_dummy (game_id, game_date, team_id, HomeTeam, H, AB, HR, K, SF,
    BB, HBP, 1B, 2B, 3B, TB, RS, RA, win_yes) AS (
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
        , tbc.finalScore
        , tbc.opponent_finalScore
        , IF(tbc.finalScore>tbc.opponent_finalScore, 1, 0) AS win_yes
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
    , IF(SUM(B.AB)=0, NULL, SUM(B.H)/SUM(B.AB)) AS AVG_HIST
    , IF(SUM(B.AB-B.K-B.HR+B.SF)=0
        , NULL
        , SUM(B.H-B.HR)/SUM(B.AB-B.K-B.HR+B.SF)) AS BABIP_HIST
    , IF(SUM(B.AB+B.BB+B.HBP+B.SF)=0
        , NULL
        , SUM(B.H+B.BB+B.HBP)/SUM(B.AB+B.BB+B.HBP+B.SF)) AS OBP_HIST
    , IF(SUM(B.AB)=0
        , NULL
        , SUM(B.TB)/SUM(B.AB)) AS SLG_HIST
    , IF(SUM(B.AB)*SUM(B.AB+B.BB+B.HBP+B.SF)=0
        , NULL
        , SUM(B.H+B.BB+B.HBP)/SUM(B.AB+B.BB+B.HBP+B.SF) + SUM(B.TB)/SUM(B.AB)
        ) AS OPS_HIST
    , IF(POW(SUM(B.RA), 2) + POW(SUM(B.RS), 2) = 0
        , 0
        , POW(SUM(B.RS), 2)
              /(POW(SUM(B.RA), 2) + POW(SUM(B.RS), 2))) AS PYEX_HIST
    , SUM(B.RS)-SUM(B.RA) AS RD_HIST
    , SUM(B.win_yes)/COUNT(*) AS WP_HIST
FROM batting_dummy A LEFT JOIN batting_dummy B
    ON A.team_id = B.team_id
        AND A.game_date > B.game_date
GROUP BY A.game_id, A.team_id
ORDER BY A.game_id, A.team_id
;

ALTER TABLE hist_tb_stats ADD PRIMARY KEY (game_id, team_id);


## Team batting rolling 100 day stats for home and away teams by game id
CREATE TEMPORARY TABLE roll_tb_stats ENGINE=MEMORY AS   -- noqa: PRS
WITH batting_dummy (game_id, game_date, team_id, HomeTeam, H, AB, HR, K, SF,
    BB, HBP, 1B, 2B, 3B, TB, RA, RS, win_yes) AS (
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
        , tbc.finalScore
        , tbc.opponent_finalScore
        , IF(tbc.finalScore>tbc.opponent_finalScore, 1, 0) AS win_yes
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
    , IF(SUM(B.AB)=0, NULL, SUM(B.H)/SUM(B.AB)) AS AVG_ROLL
    , IF(SUM(B.AB-B.K-B.HR+B.SF)=0
        , NULL
        , SUM(B.H-B.HR)/SUM(B.AB-B.K-B.HR+B.SF)) AS BABIP_ROLL
    , IF(SUM(B.AB+B.BB+B.HBP+B.SF)=0
        , NULL
        , SUM(B.H+B.BB+B.HBP)/SUM(B.AB+B.BB+B.HBP+B.SF)) AS OBP_ROLL
    , IF(SUM(B.AB)=0
        , NULL
        , SUM(B.TB)/SUM(B.AB)) AS SLG_ROLL
    , IF(SUM(B.AB)*SUM(B.AB+B.BB+B.HBP+B.SF)=0
        , NULL
        , SUM(B.H+B.BB+B.HBP)/SUM(B.AB+B.BB+B.HBP+B.SF) + SUM(B.TB)/SUM(B.AB)
        ) AS OPS_ROLL
    , IF(POW(SUM(B.RA), 2) + POW(SUM(B.RS), 2) = 0
        , 0
        , POW(SUM(B.RS), 2)
              /(POW(SUM(B.RA), 2) + POW(SUM(B.RS), 2))) AS PYEX_ROLL
    , SUM(B.RS) - SUM(B.RA) AS RD_ROLL
    , SUM(B.win_yes)/COUNT(*) AS WP_ROLL
FROM batting_dummy A LEFT JOIN batting_dummy B
    ON A.team_id = B.team_id
        AND A.game_date > B.game_date
           AND B.game_date >= DATE_SUB(A.game_date, INTERVAL 100 DAY)
GROUP BY A.game_id, A.team_id
ORDER BY A.game_id, A.team_id
;

ALTER TABLE roll_tb_stats ADD PRIMARY KEY (game_id, team_id);

## Team batting combined historic and rolling stats for home and away teams by game id
CREATE TEMPORARY TABLE tb_stats ENGINE=MEMORY AS   -- noqa: PRS
SELECT
    htb.game_id
    , htb.game_date
    , htb.team_id
    , htb.homeTeam
    , AVG_HIST
    , BABIP_HIST
    , OBP_HIST
    , SLG_HIST
    , OPS_HIST
    , PYEX_HIST
    , RD_HIST
    , WP_HIST
    , AVG_ROLL
    , BABIP_ROLL
    , OBP_ROLL
    , SLG_ROLL
    , OPS_ROLL
    , PYEX_ROLL
    , RD_ROLL
    , WP_ROLL
FROM hist_tb_stats htb JOIN roll_tb_stats rtb
    ON htb.game_id=rtb.game_id
           AND htb.team_id=rtb.team_id
ORDER BY htb.game_id
;

ALTER TABLE tb_stats ADD PRIMARY KEY (game_id, team_id);


## Team batting stats split for home and away teams by game id
CREATE TEMPORARY TABLE home_tb_stats ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM tb_stats WHERE homeTeam=1
;

ALTER TABLE home_tb_stats ADD PRIMARY KEY (game_id, team_id);


CREATE TEMPORARY TABLE away_tb_stats ENGINE=MEMORY AS  -- noqa: PRS
SELECT * FROM tb_stats WHERE homeTeam=0
;

ALTER TABLE away_tb_stats ADD PRIMARY KEY (game_id, team_id);

CREATE OR REPLACE TABLE tb_features AS
SELECT
    htb.game_id
    , htb.game_date
    , htb.AVG_HIST - atb.AVG_HIST AS TB_AVG_DIFF_HIST
    , htb.BABIP_HIST - atb.BABIP_HIST AS TB_BABIP_DIFF_HIST
    , htb.OBP_HIST - atb.OBP_HIST AS TB_OBP_DIFF_HIST
    , htb.SLG_HIST - atb.SLG_HIST AS TB_SLG_DIFF_HIST
    , htb.OPS_HIST - atb.SLG_HIST AS TB_OPS_DIFF_HIST
    , htb.PYEX_HIST - atb.PYEX_HIST AS TM_PYEX_DIFF_HIST
    , htb.RD_HIST - atb.RD_HIST AS TM_RD_DIFF_HIST
    , htb.WP_HIST - atb.WP_HIST AS TM_WP_DIFF_HIST
    , htb.AVG_ROLL - atb.AVG_ROLL AS TB_AVG_DIFF_ROLL
    , htb.BABIP_ROLL - atb.BABIP_ROLL AS TB_BABIP_DIFF_ROLL
    , htb.OBP_ROLL - atb.OBP_ROLL AS TB_OBP_DIFF_ROLL
    , htb.SLG_ROLL - atb.SLG_ROLL AS TB_SLG_DIFF_ROLL
    , htb.OPS_ROLL - atb.OPS_ROLL AS TB_OPS_DIFF_ROLL
    , htb.PYEX_ROLL - atb.PYEX_HIST AS TM_PYEX_DIFF_ROLL
    , htb.RD_ROLL - atb.RD_ROLL AS TM_RD_DIFF_ROLL
    , htb.WP_ROLL - atb.WP_ROLL AS TM_WP_DIFF_ROLL
FROM home_tb_stats htb JOIN away_tb_stats atb
    ON htb.game_id = atb.game_id
;

ALTER TABLE tb_features ADD PRIMARY KEY (game_id, game_date);

##-----------------------------------------------------------------

CREATE OR REPLACE TABLE features_table AS
SELECT
    spf.game_id
    , spf.game_date
    , spf.SP_BFP_DIFF_HIST
    , spf.SP_IP_DIFF_HIST
    , spf.SP_BB9_DIFF_HIST
    , spf.SP_HA9_DIFF_HIST
    , spf.SP_HRA9_DIFF_HIST
    , spf.SP_SO9_DIFF_HIST
    , spf.SP_SOPP_DIFF_HIST
    , spf.SP_FIP_DIFF_HIST
    , spf.SP_WHIP_DIFF_HIST
    , spf.SP_CERA_DIFF_HIST
    , spf.SP_PFR_DIFF_HIST
    , spf.SP_SWR_DIFF_HIST
    , spf.SP_OBA_DIFF_HIST
    , spf.SP_RAA_DIFF_HIST
    , spf.SP_BFP_DIFF_ROLL
    , spf.SP_IP_DIFF_ROLL
    , spf.SP_BB9_DIFF_ROLL
    , spf.SP_HA9_DIFF_ROLL
    , spf.SP_HRA9_DIFF_ROLL
    , spf.SP_SO9_DIFF_ROLL
    , spf.SP_SOPP_DIFF_ROLL
    , spf.SP_FIP_DIFF_ROLL
    , spf.SP_WHIP_DIFF_ROLL
    , spf.SP_CERA_DIFF_ROLL
    , spf.SP_PFR_DIFF_ROLL
    , spf.SP_SWR_DIFF_ROLL
    , spf.SP_OBA_DIFF_ROLL
    , spf.SP_RAA_DIFF_ROLL
    , tpf.TP_BFP_DIFF_HIST
    , tpf.TP_IP_DIFF_HIST
    , tpf.TP_BB9_DIFF_HIST
    , tpf.TP_HA9_DIFF_HIST
    , tpf.TP_HRA9_DIFF_HIST
    , tpf.TP_SO9_DIFF_HIST
    , tpf.TP_SOPP_DIFF_HIST
    , tpf.TP_FIP_DIFF_HIST
    , tpf.TP_WHIP_DIFF_HIST
    , tpf.TP_CERA_DIFF_HIST
    , tpf.TP_PFR_DIFF_HIST
    , tpf.TP_SWR_DIFF_HIST
    , tpf.TP_OBA_DIFF_HIST
    , tpf.TP_BFP_DIFF_ROLL
    , tpf.TP_IP_DIFF_ROLL
    , tpf.TP_BB9_DIFF_ROLL
    , tpf.TP_HA9_DIFF_ROLL
    , tpf.TP_HRA9_DIFF_ROLL
    , tpf.TP_SO9_DIFF_ROLL
    , tpf.TP_SOPP_DIFF_ROLL
    , tpf.TP_FIP_DIFF_ROLL
    , tpf.TP_WHIP_DIFF_ROLL
    , tpf.TP_CERA_DIFF_ROLL
    , tpf.TP_PFR_DIFF_ROLL
    , tpf.TP_SWR_DIFF_ROLL
    , tpf.TP_OBA_DIFF_ROLL
    , tbf.TB_AVG_DIFF_HIST
    , tbf.TB_BABIP_DIFF_HIST
    , tbf.TB_OBP_DIFF_HIST
    , tbf.TB_OPS_DIFF_HIST
    , tbf.TB_SLG_DIFF_HIST
    , tbf.TB_AVG_DIFF_ROLL
    , tbf.TB_BABIP_DIFF_ROLL
    , tbf.TB_OBP_DIFF_ROLL
    , tbf.TB_OPS_DIFF_ROLL
    , tbf.TB_SLG_DIFF_ROLL
    , tbf.TM_PYEX_DIFF_HIST
    , tbf.TM_PYEX_DIFF_ROLL
    , tbf.TM_RD_DIFF_HIST
    , tbf.TM_RD_DIFF_ROLL
    , tbf.TM_WP_DIFF_HIST
    , tbf.TM_WP_DIFF_ROLL
    , spf.HTWins
FROM tp_features tpf
JOIN tb_features tbf on tpf.game_id = tbf.game_id
JOIN sp_features spf ON tpf.game_id = spf.game_id
ORDER BY spf.game_id;

ALTER TABLE features_table ADD PRIMARY KEY (game_id, game_date);
