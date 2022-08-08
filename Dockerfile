FROM ubuntu:20.04 AS deps

RUN apt update \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt install -y --no-install-recommends \
  libgmp-dev \
  libmpfr-dev \
  libgsl-dev \
  libbz2-dev \
  liblzma-dev \
  libcurl4-openssl-dev \
  libssl-dev \
  make \
  python3-appdirs \
  python3-matplotlib \
  python3-pandas \
  python3-pysam \
  python3-scipy \
  python3-setuptools \
  python3-sklearn \
  python3-tqdm \
  gnuplot-nox \
  && rm -rf /var/lib/apt/lists/*

FROM deps AS builder

RUN apt update \
  && apt install -y --no-install-recommends \
  ca-certificates \
  cython3 \
  g++ \
  git \
  python3-dev \
  python3-setuptools-scm

COPY ./ /src
WORKDIR /src
RUN python3 setup.py install

FROM deps
COPY --from=builder /usr/local/ /usr/local/
COPY --from=builder /src /src
WORKDIR /mnt

ENTRYPOINT ["/usr/local/bin/smc++"]
