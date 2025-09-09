FROM continuumio/miniconda3

# Set working directory
WORKDIR /workspace

# Copy environment file if you have one, else create one
COPY environment.yml /workspace/environment.yml

# Create the conda environment named 'ads'
RUN conda env create -f environment.yml

# Activate the environment and ensure it's used by default
SHELL ["conda", "run", "-n", "ads", "/bin/bash", "-c"]

# Copy the rest of the project files
COPY . /workspace

# Set environment variables for Python and Conda
ENV PATH /opt/conda/envs/ads/bin:$PATH
ENV CONDA_DEFAULT_ENV ads

# Set entrypoint to bash so user can run scripts interactively
ENTRYPOINT ["/bin/bash"]
