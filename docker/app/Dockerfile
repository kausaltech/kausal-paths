FROM ubuntu:22.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Helsinki"

RUN mkdir /code

RUN apt-get update
RUN apt-get install -y wget

# Add the postgres repo
RUN echo "deb http://apt.postgresql.org/pub/repos/apt kinetic-pgdg main" > /etc/apt/sources.list.d/pgdg.list
RUN wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc > /etc/apt/trusted.gpg.d/pgdg.asc

RUN apt-get update
RUN apt-get install -y python3.11 libpython3.11-dev python3-pip \
	postgresql-client-common postgresql-client libpq-dev \
	gettext git
RUN apt-get install -y libsnappy-dev

RUN ln -fs /usr/bin/python3.11 /usr/bin/python3
RUN ln -fs /usr/bin/python3 /usr/bin/python
RUN ln -fs /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip

ARG EXTRA_PYPI_INDEX

# Install ipython for convenience
RUN pip install ipython

COPY requirements.txt requirements-prod.txt /code/
RUN pip install -r /code/requirements-prod.txt -r /code/requirements.txt

# Install extra dependencies
COPY requirements-kausal.txt /code/
RUN \
    if [ ! -z "$EXTRA_PYPI_INDEX" ] ; then \
        pip install --extra-index-url "${EXTRA_PYPI_INDEX}" -r /code/requirements-kausal.txt ; \
    fi

ARG MEDIA_ROOT=/srv/media
ENV MEDIA_ROOT=$MEDIA_ROOT
ARG STATIC_ROOT=/srv/static
ENV STATIC_ROOT=$STATIC_ROOT
RUN mkdir -p "$MEDIA_ROOT"

WORKDIR /code

COPY . /code/
RUN ./manage.py collectstatic --no-input
RUN ./manage.py compilemessages
COPY ./docker/app/docker-entrypoint.sh /
COPY ./docker/app/wait-for-it.sh /

ENTRYPOINT ["/bin/bash", "/docker-entrypoint.sh"]
