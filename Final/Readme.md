# Home

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Terminology](#terminology)
4. [Feature_Engineering](#featureengineering)
5. [Models](#models)
6. [Results](#results)

## Introduction

![Baseball](https://media.istockphoto.com/id/1190211599/vector/baseball-game-flat-banner-vector-template.jpg?s=612x612&w=0&k=20&c=7HdDIMrU34GievWhJqCZC_z0vEyIf0Q1XupVs4ZwBqI=)

### <a href="https://en.wikipedia.org/wiki/Baseball#History" target="_blank">Baseball</a>

The main objective of this project is to predict whether HomeTeam wins for a baseball game.
This would provide insight and analysis to fans, bettors, and analysts. By using statistical models and historical
data , predictions can help individuals make informed decisions when it comes to betting on games or simply just to
understand the potential outcome.

![Betting](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdj_OUQmcOUG9ZzbF8dipBXVPe-7fzYopW3g&usqp=CAU)

Based on historical data from 2008 to 2012, several features have been engineered using the baseball statistics
from <a href="https://en.wikipedia.org/wiki/Baseball_statistics" target="_blank">Wikipedia</a> and
<a href="https://www.mlb.com/glossary/advanced-stats" target="_blank">MLB</a>. I will discuss how these features
have been developed in detail later in this project, some performed good and most not so good. That is how it is,
creating anything that relate to our objective and testing whether it is actually good, would yield better
results.

One of the top priorities is the reproducibility of this project with the exact results that have been discussed
below. Code has been developed in Python, and Mariadb was used to play with the data, loading the dataset, and
creating features. Reproducibility is ensured by setting this entire project to run in docker. Cloning this
repository and running

`docker-compose up`

will build the required docker images, run the containers, and generate the output
inside your repo to `Backup/Output`

## Dataset

The dataset has been provided in class BDA-602 by the course instructor. Link to download the
<a href="https://teaching.mrsharky.com/data/baseball.sql.tar.gz" target="_blank">Baseball Dataset</a>.

The parent table has all info of each game that is in the data play by play. To our convenience, all the game
metrics have been grouped into different tables already in the dataset. Few tables that have been used for
our purpose in this project are `boxscore`, `game`, `pitcher_counts`, and `team_batting_counts`.

As in any dataset, this has issues too. For example, the target variable HomeTeamWins `HTWins` was supposed to be
generated from `boxscore.winner_home_or_away`.
But the data it has, is incorrect. Some rows do not match winner with their respective runs(not always greater).
Hence, target variable was generated using `boxscore.home_runs` and `boxscore.away_runs`. Only when home team runs
are greater, it was considered that home team won or 1, rest of the conditions equal or less is considered 0.

The data in columns about Caught Stealings and Stolen bases in all tables are all zeroes. They could be built from
innings table, but any features that require these columns are avoided in my project.

There are duplicate columns like forceout & force_out, flyout & fly_out, and lineout & line_out. They should be
combined(added) to use in any feature. As sometimes data is missing in forceout, only has zeroes, but same rows have
data in force_out, and rest of rows are all zeroes where the former had data. Hence, combining them would be
good approach. Similarly, for other columns mentioned. I have highlighted more mistakes in the data in the coming
sections, wherever applicable.

## Terminology

## Feature_Engineering

## Models

## Results
