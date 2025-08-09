FROM gcc:12.5

# Expect to run Python scripts
RUN apt update -y \
    && apt upgrade -y \
    && apt install -y python3

# Makefiles expect gcc and g++ to be in /usr/bin
RUN ln -s /usr/local/bin/gcc /usr/bin/gcc
RUN ln -s /usr/local/bin/g++ /usr/bin/g++
