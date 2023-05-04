#!/bin/bash

sleep 20

if mysql -h vij_mariadb -P 3306 -u root -px11docker -e "USE baseball;"
then
  echo "Baseball Database exists, Using the database"
  echo "Executing the SQL Script to create baseball feature tables"
  mysql -h vij_mariadb -P 3306 -u root -px11docker baseball < features.sql
  echo "Finished creating required tables"
else
  echo "Baseball Database does not exist, Proceeding to Create"
  mysql -h vij_mariadb -P 3306 -u root -px11docker -e "CREATE DATABASE baseball"
  echo "Baseball Database created"
  echo "Finding baseball.sql from the root of the project"
  echo "Setting up the database now"
  mysql -h vij_mariadb -P 3306 -u root -px11docker baseball < baseball.sql
  echo "Executing the SQL Script to create baseball feature tables"
  mysql -h vij_mariadb -P 3306 -u root -px11docker baseball < features.sql
  echo "Finished creating required tables"
fi

python3 main.py
