# Artifact Appendix

Paper title: **Making Sense of Private Advertising: A Principled Approach to a Complex Ecosystem**

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [x] **Reproduced**

## Description

```bibtex
@Article{PoPETS:HCKVD26,
  author    =   "Kyle Hogan and
                 Alishah Chator and
                 Gabriel Kaptchuk and
                 Mayank Varia and
                 Srinivas Devadas",
  title     =   "{Making Sense of Private Advertising: A Principled Approach to a Complex Ecosystem}",
  year      =   2026,
  volume    =   2026,
  journal   =   "{Proceedings on Privacy Enhancing Technologies}",
}
```

We provide an implementation of our ideal functionalities and generic parameterizing functions as well as an empirical distinguishing game. 
The distinguishing game computes the sample complexity required to distinguish two sample distributions with varying parameters for the utility of our functionalities and varying distances between the two distributions. 

### Security/Privacy Issues and Ethical Concerns

N/A

## Basic Requirements

### Hardware Requirements 

Can be run on commodity hardware, e.g. a laptop. However, the distinguishing game is parallelized and will run faster on a server class machine.

### Software Requirements

Experiments were run on Ubuntu 24.04 and have not been tested on Windows or MacOS.

We provide a [Dockerfile](./Dockerfile) (Docker version 28.4.0) and [environment.yml](./environment.yml) (conda 23.1.0) to manage the installation of required packages.


### Estimated Time and Storage Consumption 

No substantial storage requirements (<5GB)
All experiments are computationally intensive and can take tens to hundreds of hours for larger numbers of trials. We provide a "small" option to run experiments with 10 trials as a test.

We also include example datasets in the `plots/` directory for reproducing the plots from the paper without re-running the experiments.

## Environment 

### Accessibility 

https://github.com/kylehogan/idealAdsFunctionalities/tree/main

### Set up the environment

On a machine running Ubuntu 20.04, 22.04, or 24.04:

First ensure that docker and git are installed.

Then, clone the repo and build the docker container:

```bash
git clone --recurse-submodules https://github.com/kylehogan/idealAdsFunctionalities.git
cd idealAdsFunctionalities/
docker build -t idealads .
```

### Testing the Environment

To test the installation, first run the docker container:

```bash
docker run -it idealads
```

To test functionality, first reproduce one of the plots from the paper using the provided data.

Inside the Docker container, run:

```bash
 ./reproduce_plots.sh --plots-only private_v_nonprivate
```

This should reproduce the "Private vs. Nonprivate" plot as it appears in the paper and save it as `plots/private_v_nonprivate/pval_left_private_v_nonprivate.png`

![private vs. nonprivate distinguishing game](plots/private_v_nonprivate/pval_left_private_v_nonprivate.png "Private vs. Nonprivate Distinguishing Game")

Then, run the "small" version of one of the plots. This will only run 10 trials and should produce a plot that is not missing any lines or datapoints and generally increases in sample complexity as the alpha-epsilon value and total variation distance decrease. You can adjust the number of cores with `--cores N` (requires approximately 15-20 minutes with 8 cores).

```bash
 ./reproduce_plots.sh --clean --small engagement
```

## Artifact Evaluation

This section should include all the steps required to evaluate your artifact's
functionality and validate your paper's key results and claims. Therefore,
highlight your paper's main results and claims in the first subsection. And
describe the experiments that support your claims in the subsection after that.

### Main Results and Claims

List all your paper's results and claims that are supported by your submitted
artifacts.

#### Main Result 1: Name

Describe the results in 1 to 3 sentences. Mention what the independent and
dependent variables are; independent variables are the ones on the x-axes of
your figures, whereas the dependent ones are on the y-axes. By varying the
independent variable (e.g., file size) in a given manner (e.g., linearly), we
expect to see trends in the dependent variable (e.g., runtime, communication
overhead) vary in another manner (e.g., exponentially). Refer to the related
sections, figures, and/or tables in your paper and reference the experiments
that support this result/claim. See example below.

#### Main Result 2: Example Name

Our paper claims that when varying the file size linearly, the runtime also
increases linearly. This claim is reproducible by executing our
[Experiment 2](#experiment-2-example-name). In this experiment, we change the
file size linearly, from 2KB to 24KB, at intervals of 2KB each, and we show that
the runtime also increases linearly, reaching at most 1ms. We report these
results in "Figure 1a" and "Table 3" (Column 3 or Row 2) of our paper.

### Experiments
List each experiment to execute to reproduce your results. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes to execute in human and compute times (approximately).
 - How much space it consumes on disk (approximately) (omit if <10GB).
 - Which claim and results does it support, and how.

#### Experiment 1: Private vs. Nonprivate Sample Complexity
- Time: replace with estimate in human-minutes/hours + compute-minutes/hours.
- Storage: replace with estimate for disk space used (omit if <10GB).

Provide a short explanation of the experiment and expected results. Describe
thoroughly the steps to perform the experiment and to collect and organize the
results as expected from your paper (see example below). Use code segments to
simplify the workflow, as follows.

```bash
python3 experiment_1.py
```

#### Experiment 2: Example Name

- Time: 10 human-minutes + 3 compute-hours
- Storage: 20GB

This example experiment reproduces
[Main Result 2: Example Name](#main-result-2-example-name), the following script
will run the simulation automatically with the different parameters specified in
the paper. (You may run the following command from the example Docker image.)

```bash
python3 main.py
```

Results from this example experiment will be aggregated over several iterations
by the script and output directly in raw format along with variances and
standard deviations in the `output-folder/` directory. You will also find there
the plots for "Figure 1a" in `.pdf` format and the table for "Table 3" in `.tex`
format. These can be directly compared to the results reported in the paper, and
should not quantitatively vary by more than 5% from expected results.


## Limitations 

The runtime for reproducing Experiment 1 and Experiment 2 can be quite long, even with a substantial number of cores. Due to the randomness of our experiments there is high variability in the number of samples required for distinguishing and the curves on our plots do not smooth out until around 100 trials for Experiments 1 and 2 and 1000 trials for experiments 3 and 4. 

Fully reproducing these plots will likely take more time/cores than most people will find reasonable to dedicate. 

## Notes on Reusability 

While we used basic parameterizing functionalities and simple distributions in evaluating this work, both aspects are highly customizable.

For example, an alternative targeting parameterizing functionalities could be a click predictor model.

Additionally, more complex distributions over users could incorporate correlations over specific features of interest and align with the use of those features during targeting or engagement.