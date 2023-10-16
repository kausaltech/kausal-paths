# syntax=docker/dockerfile:1
FROM ubuntu:23.04 AS base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Helsinki"
ENV PYTHON_VERSION=3.11

RUN mkdir /code

RUN apt-get update
RUN apt-get install -y wget

# Add the postgres repo
RUN echo "deb http://apt.postgresql.org/pub/repos/apt lunar-pgdg main" > /etc/apt/sources.list.d/pgdg.list
RUN wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc > /etc/apt/trusted.gpg.d/pgdg.asc

# Set up APT package caching
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y \
    python${PYTHON_VERSION} libpython${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python3-pip \
	postgresql-client-common postgresql-client libpq-dev \
	gettext git libsnappy-dev

RUN /usr/bin/python${PYTHON_VERSION} -m venv /venv
ENV PATH="/venv/bin:${PATH}"
ENV XDG_CACHE_HOME=/cache
RUN mkdir /cache

# Install also ipython for convenience
RUN --mount=type=cache,target=${XDG_CACHE_HOME}/pip \
    python -m pip install --upgrade pip ipython

COPY requirements.txt requirements-prod.txt /code/
RUN --mount=type=cache,target=${XDG_CACHE_HOME}/pip \
    pip install -r /code/requirements-prod.txt -r /code/requirements.txt

ARG EXTRA_PYPI_INDEX

# Install extra dependencies
COPY requirements-kausal.txt /code/
RUN --mount=type=cache,target=${XDG_CACHE_HOME}/pip \
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
