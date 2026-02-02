# SISAP 2026 Challenge: Working example in Python 

This repository is a working example for the SISAP 2026 Indexing Challenge <https://sisap-challenges.github.io/>, working with Python and GitHub Actions.

## Steps for running
It requires a working installation of Python 3.10 with conda support, for example using Anaconda. 

The steps are the following:

1. Clone the example repository
1. Run
1. Evaluate

The full set of installation instructions are listed in the GitHub Actions workflow

<https://github.com/sisap-challenges/sisap26-python-baseline/blob/main/.github/workflows/ci.yml>

Note that you will need to adjust your scripts to hold the correct hyperparameters for any benchmark you use.

### Clone this repository
```base
git clone https://github.com/sisap-challenges/sisap26-python-baseline
cd sisap26-example-python
```

### Run
Run the tasks on an example input using

```bash
python search.py --task {task1, task2}
```

It will automatically take care of downloading the necessary example dataset.



### Evaluation

```bash
python eval.py results.csv
```
will produce a summary file of the results with the computed recall against the ground truth data. 

This csv file can be further processed to create plots (using `python plot.py --task {task1, task2}`) and show the fastest solutions above a certain recall threshold (using `python show_operating_points.py`).

## How to take this to create my own system
You can fork this repository and polish it to create your solution. Please also take care of the ci workflow (see below).

## GitHub Actions: Continuous integration 

You can monitor your runnings in the "Actions" tab of the GitHub panel: for instance, you can see some runs of this repository:
<https://github.com/sisap-challenges/sisap26-python-baseline/actions>

 
