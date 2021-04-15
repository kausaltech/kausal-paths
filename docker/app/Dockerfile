FROM ubuntu:20.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ARG MEDIA_ROOT
ENV MEDIA_ROOT $MEDIA_ROOT
ARG STATIC_ROOT
ENV STATIC_ROOT $STATIC_ROOT

RUN mkdir /code
RUN mkdir -p "$MEDIA_ROOT"

RUN apt-get update

# Install PostgreSQL 13
RUN apt-get install -y gnupg wget lsb-release software-properties-common
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 apt-key add -
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -sc)-pgdg main" > /etc/apt/sources.list.d/postgres.list
RUN apt-get update
RUN apt-get install -y python3.9 python3-pip libpython3.9-dev \
	postgresql-client-common postgresql-client libpq-dev \
	gettext git-core

RUN ln -fs /usr/bin/python3.9 /usr/bin/python3
RUN ln -fs /usr/bin/python3 /usr/bin/python
RUN ln -fs /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip
COPY requirements.txt requirements-prod.txt /code/
RUN pip install -r /code/requirements-prod.txt -r /code/requirements.txt

# Install ipython for convenience
RUN pip install ipython

WORKDIR /code

COPY . /code/
RUN ./manage.py collectstatic --no-input
RUN ./manage.py compilemessages
COPY ./docker/app/docker-entrypoint.sh /
COPY ./docker/app/wait-for-it.sh /

ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]
