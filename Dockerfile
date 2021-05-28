FROM continuumio/miniconda3

RUN conda config --add channels defaults
RUN conda config --add channels bioconda
RUN conda config --add channels conda-forge

RUN conda install pytorch biopython scikit-learn

ADD . .
