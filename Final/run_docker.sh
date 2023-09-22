#!/bin/bash

# Setup Health Check for mariadb to delay and start python container
# Sleep is not needed anymore.
# sleep 40

# Sources:
# https://stackoverflow.com/questions/33170489/how-can-i-access-my-docker-maria-db
# https://stackoverflow.com/questions/8940230/how-to-run-sql-script-in-mysql

echo "Checking if Baseball Database Exists!!"

if  mysql -h vij_mariadb -P 3306 -u root -px11docker -e "USE baseball;" 2> /dev/null
then
  echo "Baseball Database exists, Using the database"
  echo "Executing the SQL Script to create baseball feature tables"
  mysql -h vij_mariadb -P 3306 -u root -px11docker baseball < features.sql
  echo "Finished creating required tables"
else
  echo "Baseball Database does not exist, Proceeding to Create"
  mysql -h vij_mariadb -P 3306 -u root -px11docker -e "CREATE DATABASE baseball"
  echo "Baseball Database created"
  echo "Finding baseball.sql file"
  echo "Setting up the database now, please wait..."
  mysql -h vij_mariadb -P 3306 -u root -px11docker baseball < baseball.sql
  echo "Executing the SQL Script to create baseball feature tables now..."
  mysql -h vij_mariadb -P 3306 -u root -px11docker baseball < features.sql
  echo "Finished creating required tables"
fi

echo "Running python files now..."
python3 main.py
echo "Done!"
echo "Results are saved to BDA-602/Backup/Output"
