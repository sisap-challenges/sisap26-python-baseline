# SISAP 2026 Challenge: Working example in Python 

This repository is a working example for the SISAP 2026 Indexing Challenge <https://sisap-challenges.github.io/>, working with Python and GitHub Actions.

## Installation & Setup

### 1. Clone this repository
```bash
git clone https://github.com/sisap-challenges/sisap26-python-baseline
cd sisap26-python-baseline
```

### 2. Install Dependencies
This repository requires Python 3.9+ and several dependencies. We provide a helper script for easy setup, or you can install manually.

#### Option A: Quick Start (Linux/Mac)
Use the provided install script to set up a virtual environment and install dependencies (including CPU-optimized PyTorch):

```bash
chmod +x install.sh
./install.sh
source venv/bin/activate
```

#### Option B: Manual Installation
1. Install base requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Install CPU-only PyTorch (to avoid large CUDA downloads):
   ```bash
   pip install torch~=2.4.0 --index-url https://download.pytorch.org/whl/cpu
   ```

#### Option C: Docker
Build and run using Docker:
```bash
docker build -t sisap-baseline .
```

## Running the Code
The suggested approach is to run the the Docker container as detailed in `run_search.sh`.

### Evaluation

```bash
python eval.py results.csv
```
will produce a summary file of the results with the computed recall against the ground truth data. 

This csv file can be further processed to create plots (using `python plot.py --task {task1, task2, task3} res.csv`) and show the fastest solutions above a certain recall threshold (using `python show_operating_points.py`).

## Task configuration format (`config.json`)

Each dataset directory under `data/` contains a `config.json` file that describes the task. The fields are:

| Field | Type | Description |
|---|---|---|
| `task` | string | Task identifier: `"task1"`, `"task2"`, or `"task3"` |
| `data` | string | HDF5 group containing the database vectors (e.g. `"train"`) |
| `queries` | string | HDF5 path to the query vectors (task2/task3 only) |
| `gt_I` | string or array | HDF5 path(s) to the ground-truth nearest-neighbor indices |
| `k` | int | Number of nearest neighbors to retrieve |
| `dataset_name` | string | Human-readable dataset identifier |
| `filename` | string | Name of the HDF5 data file |
| `sparse` | bool | If `true`, vectors are sparse (task3 only); absent means `false` |

**Example (task1):** all-kNN — no separate query set; `gt_I` is a list of two HDF5 paths `["allknn", "knns"]` pointing to the full neighbor graph.

**Example (task2/3):** query-search — `queries` and `gt_I` are single HDF5 paths for the query vectors and their ground-truth neighbors, respectively.

## How to take this to create my own system
You can fork this repository and polish it to create your solution. Please also take care of the ci workflow (see below).

## GitHub Actions: Continuous integration 

You can monitor your runnings in the "Actions" tab of the GitHub panel: for instance, you can see some runs of this repository:
<https://github.com/sisap-challenges/sisap26-python-baseline/actions>

 
