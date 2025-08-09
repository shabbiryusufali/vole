FROM ubuntu:22.04

RUN RUN apt update -y \
    && apt upgrade -y \
    && apt install -y \
    build-essential \
    clang \
    git \
    python3 \
    wget \
    curl \
    ca-certificates \
    make

RUN git clone https://github.com/CyberGrandChallenge/cb-testing.git /usr/share/cb-testing
