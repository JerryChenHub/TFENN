FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90

ARG MINICONDA_INSTALLER=Miniconda3-py312_26.5.3-2-Linux-x86_64.sh
ARG MINICONDA_SHA256=37606f9f03ced8ef60f4ffc76b21dda01728eac8a632dcab316c891cea4fe2f5
ARG OPLS_REF=319521f5551782c7f9016a51f84225578e583068
ARG OPLS_ARCHIVE_SHA256=352f29c867de33aa22926d9d8117c2ce26c4c4f023d7360303982bf7eea8fc89

LABEL org.opencontainers.image.title="TFENN_A"
LABEL org.opencontainers.image.source="https://github.com/JerryChenHub/TFENN"
LABEL org.opencontainers.image.description="TFENN development and test environment"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        git \
        less \
        procps \
        vim-tiny \
    && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --silent --show-error \
        "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" \
        --output /tmp/miniconda.sh \
    && echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum --check \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh

ENV PATH="/opt/miniconda/bin:${PATH}"

COPY environment.yml /tmp/environment.yml

RUN conda config --system --set channel_priority strict \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda env create --file /tmp/environment.yml \
    && conda clean --all --yes \
    && rm /tmp/environment.yml

ENV CONDA_DEFAULT_ENV=tfenn
ENV CONDA_PREFIX=/opt/miniconda/envs/tfenn
ENV PATH="/opt/miniconda/envs/tfenn/bin:/opt/miniconda/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /Project

RUN curl --fail --location --silent --show-error \
        --retry 5 --retry-all-errors --connect-timeout 20 --max-time 300 \
        "https://codeload.github.com/JerryChenHub/OPLS2020_Static/tar.gz/${OPLS_REF}" \
        --output /tmp/opls.tar.gz \
    && echo "${OPLS_ARCHIVE_SHA256}  /tmp/opls.tar.gz" | sha256sum --check \
    && mkdir /Project/OPLS2020_Static \
    && tar --extract --gzip --file /tmp/opls.tar.gz \
        --strip-components=1 --directory /Project/OPLS2020_Static \
    && rm /tmp/opls.tar.gz \
    && python -m pip install --no-build-isolation --no-deps --editable /Project/OPLS2020_Static

WORKDIR /Project/TFENN

COPY . .

RUN python -m pip install --no-deps --editable . \
    && python -m pip check \
    && python docker/smoke_test.py

CMD ["/bin/bash"]
