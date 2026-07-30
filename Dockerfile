# syntax=docker/dockerfile:1
FROM ubuntu:24.04

ARG MINICONDA_INSTALLER=Miniconda3-py312_26.5.3-2-Linux-x86_64.sh
ARG MINICONDA_SHA256=37606f9f03ced8ef60f4ffc76b21dda01728eac8a632dcab316c891cea4fe2f5
ARG OPLS_REPOSITORY=https://github.com/JerryChenHub/OPLS2020_Static.git
ARG OPLS_REF=a4dbe85edfd25fc9b4c2cf5ac2109f318455f4e1

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

RUN git clone --filter=blob:none "${OPLS_REPOSITORY}" OPLS2020_Static \
    && git -C /Project/OPLS2020_Static checkout --detach "${OPLS_REF}" \
    && test "$(git -C /Project/OPLS2020_Static rev-parse HEAD)" = "${OPLS_REF}" \
    && python -m pip install --no-build-isolation --no-deps --editable /Project/OPLS2020_Static

WORKDIR /Project/TFENN

COPY . .

RUN python -m pip install --no-deps --editable . \
    && python -m pip check \
    && python docker/smoke_test.py

CMD ["/bin/bash"]
