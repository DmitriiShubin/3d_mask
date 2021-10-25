#FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
FROM  python:3.7.12-slim-buster

COPY . /
#COPY ./requirements.txt .
#COPY ./docker-entrypoint.sh .
RUN pip install -r requirements.txt


CMD ['bash','docker-entrypoint.sh']



