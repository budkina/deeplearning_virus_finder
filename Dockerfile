FROM continuumio/miniconda3

RUN conda config --add channels defaults
RUN conda config --add channels bioconda
RUN conda config --add channels conda-forge

RUN conda install pytorch

RUN conda install biopython
RUN conda install gensim

RUN conda install scikit-learn

ADD . .
