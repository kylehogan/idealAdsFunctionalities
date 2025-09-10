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
 ./reproduce_plots.sh --small engagement
```

## Artifact Evaluation


### Main Results and Claims

#### Main Result 1: Relating advertising privacy protections to increased ad campaign size

The main result of our empirical evaluation is to demonstrate that adding privacy protections to an advertising ecosystem increases the campaign size required to learn information about those campaign's audiences, but does not _prevent_ this learning.

We represent this by several parameters for our advertising functionalities that allow us to tune the impact of privatizing that functionality along with a distinguishing game over the ads ecosystem run on different underlying user distributions.

Decreasing the total variation distance (x-axis) of the underling user distributions naturally increases the sample complexity (y-axis) of distinguishing the output of an advertising campaign run over those distributions. 

First, we show the impact of of privatizing advertising by combining these parameters in [Experiment 1](#experiment-1-private-vs-nonprivate-sample-complexity).

We then show the impact of differential privacy at different epsilon values on the sample complexity of distinguishing in [Experiment 2](#experiment-2-differentially-private-metrics).

We also run several experiments demonstrating the additional impact of reducing targeting and engagement accuracy by decreasing our alpha-engagement in [Experiment 3](#experiment-3-engagement) and alpha-targeting parameters in [Experiment 4](#experiment-4-private-targeting).

### Experiments
 We list the methods to produce each individual plot from the paper below, but all the plots can be reproduced together as well. The default number of cores is 8 and can be changes using the `--cores N` flag. 
 
 Note that [Experiment 1](#experiment-1-private-vs-nonprivate-sample-complexity) and [Experiment 2](#experiment-2-differentially-private-metrics) are particularly computationally intensive and can take several days with approximately 60 cores.

 None of the experiments require more than a few minutes of human involvement.

 Each experiment outputs a dataset `combined.parquet`, metadata for that dataset `combined_metadata.parquet`, and a plot `pval_left_[name].png` in the corresponding folder in the `plots/` directory. 
 The focus of this work is the trends in the distinguishing complexity more so than the exact sample complexity values and the trends produced by these experiments should match those in the paper (unless produced with the `--small` flag).

To rerun all experiments and reproduce plots as shown in the paper:
 ```bash
./reproduce_plots.sh 
```

To reproduce plots as shown in the paper using the data from the paper, without rerunning the experiments:
 ```bash
./reproduce_plots.sh --plots-only
```

To produce "small" versions of all plots by rerunning the experiments for a minimal number of trials (10 trials):
 ```bash
./reproduce_plots.sh --small
```

[Experiment 1: Private vs. Nonprivate Sample Complexity](#experiment-1-private-vs-nonprivate-sample-complexity)
#### Experiment 1: Private vs. Nonprivate Sample Complexity
This experiment reproduces Figure 8 from the paper and compares the sample complexity of our distinguishing game on an example private vs. non-private advertising ecosystem to the baseline sample complexity of directly distinguishing the two underlying distributions.

You can adjust the number of cores with `--cores N` (default is 8). It is computationally intensive and will by default run 100 trials, taking several days on 60 cores. 
The `--small` flag can be used to run only 10 trials and the `--plots-only` flag can be used to reproduce the plot using the example data without rerunning the experiment.

To run: 

```bash
./reproduce_plots.sh private_v_nonprivate
```

#### Experiment 2: Differentially Private Metrics

This experiment reproduces Figure 10 from the paper and demonstrates the impact of decreasing the epsilon value for the differentially private metrics functionality.

You can adjust the number of cores with `--cores N` (default is 8). It is computationally intensive and will by default run 100 trials, taking several days on 60 cores. 
The `--small` flag can be used to run only 10 trials and the `--plots-only` flag can be used to reproduce the plot using the example data without rerunning the experiment.

To run: 

```bash
./reproduce_plots.sh epsilon
```

#### Experiment 3: Engagement

This experiment reproduces Figure 11 from the paper and demonstrates the impact of decreasing the alpha-engagement parameter, or how likely users are to engage with ads at all. This is _not_ a privacy parameter, but allows for tuning click rates to match desired values, e.g. from a specific ads dataset.

You can adjust the number of cores with `--cores N` (default is 8). It is computationally intensive and will by default run 100 trials, taking several days on 60 cores. 
The `--small` flag can be used to run only 10 trials and the `--plots-only` flag can be used to reproduce the plot using the example data without rerunning the experiment.

To run: 

```bash
./reproduce_plots.sh engagement
```

#### Experiment 4: Private Targeting

This experiment reproduces Figure 12 from the paper and demonstrates the impact of decreasing the alpha-targeting parameter, or how likely users are to receive a more relevant ad over a less relevant one. It represents the "accuracy" of targeting and private targeting functionalities will tend to be less likely to show the most relevant ad due to restricted or noisy user data.

You can adjust the number of cores with `--cores N` (default is 8). It is computationally intensive and will by default run 100 trials, taking several days on 60 cores. 
The `--small` flag can be used to run only 10 trials and the `--plots-only` flag can be used to reproduce the plot using the example data without rerunning the experiment.

To run: 

```bash
./reproduce_plots.sh targeting
```

## Limitations 

The runtime for reproducing Experiment 1 and Experiment 2 can be quite long, even with a substantial number of cores. Due to the randomness of our experiments there is high variability in the number of samples required for distinguishing and the curves on our plots do not smooth out until around 100 trials for Experiments 1 and 2 and 1000 trials for experiments 3 and 4. 

Fully reproducing these plots will likely take more time/cores than most people will find reasonable to dedicate. 

## Notes on Reusability 

While we used basic parameterizing functionalities and simple distributions in evaluating this work, both aspects are highly customizable.
For example, an alternative targeting parameterizing functionalities could be a click predictor model.

Additionally, more complex distributions over users could incorporate correlations over specific features of interest and align with the use of those features during targeting or engagement.

Finally, while we provide `reproduce_plots.sh` to run `distinguishing_game.py` and produce the plots as they appear in the paper, `distinguishing_game.py` accept more arguments than are supported by `reproduce_plots.sh` and could be used to run larger numbers of trials or different total variation distances, etc.