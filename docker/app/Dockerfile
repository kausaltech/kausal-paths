ARG base_image=debian:testing-slim

#
# Invocations common for builder and final stages
#
FROM ${base_image} AS base

ARG runtime_deps="libpq5 postgresql-client gettext \
    libproj25 libgdal34 libjpeg62-turbo libtiff6 libxml2 libffi8 libxslt1.1 \
    libwebp7 libvoikko1 voikko-fi curl libpcre3 \
    libpython3.12t64 python3.12 python3.12-venv python3-pip-whl \
    iputils-ping inetutils-telnet redis-tools restic procps git"
ARG build_time_deps="libpq-dev build-essential \
    zlib1g-dev libjpeg-dev libtiff-dev libopenjp2-7-dev libwebp-dev \
    binutils libproj-dev libgdal-dev \
    libxml2-dev libxslt1-dev libffi-dev libpython3.12-dev"
# ARG node_install_url="https://deb.nodesource.com/setup_lts.x"
ARG app_path=/code
ARG app_user_uid=1000
ARG app_user_gid=1000

ENV VIRTUAL_ENV=/venv
ENV PATH=/venv/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Helsinki

RUN mkdir -p ${app_path}
# Set up APT package caching
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update && apt upgrade -y && apt install --no-install-recommends -y \
    ${runtime_deps} && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
#  && \
#  (curl -fsSL "${node_install_url}" | bash) && \
#  apt install --no-install-recommends -y nodejs


#
# Builder stage
#
FROM base AS builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update && apt install --no-install-recommends -y \
	${runtime_deps} ${build_time_deps}

# Create python virtual env
RUN python -m venv ${VIRTUAL_ENV}

ENV XDG_CACHE_HOME=/cache

COPY requirements.txt requirements-prod.txt ${app_path}/
RUN --mount=type=cache,target=${XDG_CACHE_HOME}/pip \
    pip install -r ${app_path}/requirements-prod.txt -r ${app_path}/requirements.txt

# Install extra dependencies
COPY requirements-kausal.txt ${app_path}/
RUN --mount=type=secret,id=EXTRA_PYPI_INDEX \
    if [ -f /run/secrets/EXTRA_PYPI_INDEX ] ; then \
        pip install --extra-index-url "$(cat /run/secrets/EXTRA_PYPI_INDEX)" -r ${app_path}/requirements-kausal.txt ; \
    fi

# COPY package.json package-lock.json ${app_path}/
WORKDIR ${app_path}
# RUN npm install

ARG INSTALL_DEV_DEPS
COPY requirements-dev.txt ${app_path}/
RUN --mount=type=cache,target=${XDG_CACHE_HOME}/pip \
    if [ ! -z "${INSTALL_DEV_DEPS}" ] ; then ${VENV_PATH}/bin/pip install -r ${app_path}/requirements-dev.txt ; fi

#
# Final image
#
FROM base AS final

ARG BUILD_ID
ENV BUILD_ID ${BUILD_ID}
ARG SENTRY_RELEASE ${SENTRY_RELEASE}

ENV VENV_PATH=/venv
ENV PATH=${VENV_PATH}/bin:${PATH}

COPY --from=builder /venv /venv
# COPY --from=builder ${app_path}/node_modules ${app_path}/node_modules
COPY . ${app_path}/

RUN mkdir /scripts
COPY ./docker/app/*.sh /scripts/
COPY ./docker/app/uwsgi.ini /
RUN chmod a+x /scripts/*.sh

WORKDIR ${app_path}

ARG STATIC_ROOT=/srv/static
ENV STATIC_ROOT ${STATIC_ROOT}

ARG MEDIA_ROOT=/srv/media
ENV MEDIA_ROOT ${MEDIA_ROOT}

RUN mkdir -p ${MEDIA_ROOT} ${STATIC_ROOT}

RUN groupadd -g ${app_user_gid} user && useradd --no-log-init -m -d /home/user -g ${app_user_gid} -u ${app_user_uid} -r user
RUN chown -R user:user /code /srv/media /srv/static

# Switch to the app user
USER user

RUN ./manage.py collectstatic --no-input
RUN ./manage.py compilemessages
# RUN pybabel compile -D notifications -d locale

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV UWSGI_MODULE paths.wsgi

EXPOSE 8000/tcp
ENTRYPOINT ["/bin/bash", "/scripts/docker-entrypoint.sh"]
