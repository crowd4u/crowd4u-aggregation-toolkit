# Base image
FROM python:3.10-bullseye

# Set environment variable for CmdStanPy version
ENV CMDSTANPY_VERSION=1.2.5

# Install required packages including python2 and curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
# Install CmdStanPy and dependencies
RUN pip3 install matplotlib==3.7.1 numpy==1.24.3 scipy==1.10.1 \
                sympy==1.11.1 pandas==2.0.1 arviz==0.20.0 \
                jupyterlab==4.3.4 ipykernel==6.22.0 nest-asyncio==1.6.0 \
                ipywidgets==8.1.8 crowd-kit==1.3.0.post0

RUN pip3 install cmdstanpy[all]==$CMDSTANPY_VERSION
RUN python3 -c 'import cmdstanpy; cmdstanpy.install_cmdstan(version="2.32.2")'

# Set working directory
WORKDIR /toolkit_root
