FROM continuumio/miniconda3

# Set working directory
WORKDIR /workspace

# Copy the project files
COPY . /workspace

RUN chmod +x reproduce_plots.sh

RUN conda env update --file ./environment.yml && \
    conda clean -tipy

# Set entrypoint to bash so user can run scripts interactively
ENTRYPOINT ["/bin/bash"]



