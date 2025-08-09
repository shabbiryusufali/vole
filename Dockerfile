FROM ubuntu:22.04

RUN apt update -y \
    && apt upgrade -y \
    && apt install -y software-properties-common \
        build-essential \
        python3 \
        gcc-10 g++-10 \
        gcc-11 g++-11 \
        gcc-12 g++-12 \
        clang-12 clang++-12 \
        clang-13 clang++-13 \
        clang-14 clang++-14

ENV CC=gcc-12
ENV CXX=g++-12