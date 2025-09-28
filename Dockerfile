FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y espeak ffmpeg g++ gcc git sox

COPY . /opt/spellcaster

RUN python -m pip install --upgrade pip setuptools wheel