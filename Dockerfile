FROM python:3.10.6-bullseye

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Source for Mariadb Connector/C Version incompatibility issues
# https://community.home-assistant.io/t/unable-to-load-the-mariadb-module/459791/2
# Source for mariadb_repo_setup permission issue
# https://www.shells.com/l/en-US/tutorial/How-to-Fix-Shell-Script-Permission-Denied-Error-in-Linux
# Installing mariadb-client also, as running the script gives mysql command not found error
# https://stackoverflow.com/questions/53219709/docker-mysql-command-not-found-in-sh-file

RUN wget https://downloads.mariadb.com/MariaDB/mariadb_repo_setup \
    && chmod u+x mariadb_repo_setup \
    && ./mariadb_repo_setup --mariadb-server-version="mariadb-10.6"


# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     libmariadb3 \
     libmariadb-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy over code
# Copy baseball.sql from root of my project dir
COPY baseball.sql $APP_HOME
COPY Final $APP_HOME
