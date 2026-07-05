#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import csv
import glob
import json
from datasets import DATASETS, get_fn, prepare, get_h5_item


def get_all_results(dirname):
    """Return all paths with result .h5 file under dirname."""
    seen = set()
    mask = dirname + "/**/*.h5"
    print(f"Searching for results matching: {mask}")
    ret = []
    for fn in glob.iglob(mask, recursive=True):
        if fn in seen:
            continue
        seen.add(fn)
        try:
            with h5py.File(fn, "r") as f:
                if "knns" not in f:
                    print(f"Ignoring {fn}")
                    continue

            ret += [fn]
        except:
            print(f"Could not load {fn}")

    return ret


def load_results(fn):
    with h5py.File(fn, "r") as f:
        return dict(f.attrs), np.array(f["knns"])


def get_recall(I, gt, k):
    """
    Compute Recall@k averaged over all queries.
    
    Handles two result formats:
    - n x k: results are expected to exclude self-loops
    - n x (k+1): results are expected to include self-loops in first position
    
    Ground truth format is assumed to be n x m where m >= k+1,
    with self-loops in the first column.
    
    Always compares the k nearest neighbors EXCLUDING self-loops.
    """
    assert I.shape[0] == gt.shape[0], "query count mismatch between results and ground truth"
    assert gt.shape[1] >= k + 1, f"Ground truth needs at least {k+1} columns (self + {k} neighbors), got {gt.shape[1]}"
    
    if I.shape[1] == k + 1:
        # Results include self-loops: use columns [1:k+1] (skip first column)
        I_to_compare = I[:, 1:k+1]
        print(f"  Results shape {I.shape}: assumed self-loops, comparing columns [1:{k+1}]")
    elif I.shape[1] == k:
        # Results exclude self-loops: use columns [:k]
        I_to_compare = I[:, :k]
        print(f"  Results shape {I.shape}: assumed no self-loops, comparing columns [0:{k}]")
    else:
        raise ValueError(f"Results shape {I.shape} is neither {I.shape[0]}x{k} nor {I.shape[0]}x{k+1}")
    
    # Ground truth: always skip first column (self-loop) and use next k columns
    gt_to_compare = gt[:, 1:k+1]  # Skip self-loop, take next k neighbors
    
    hits = np.array([
        len(np.intersect1d(I_to_compare[i], gt_to_compare[i]))
        for i in range(len(I))
    ])
    return hits.sum() / (len(I) * k)


def add_details_from_tira(fn, row):
    software_details = Path(fn).parent.parent / "execution-details.json"

    if software_details.is_file():
        software_details = json.loads(software_details.read_text())["system"]
        to_delete = ["docker_software_id", "user_image_name", "tira_image_name", "cache_behaviour", "task_id", "paper_link", "input_docker_software", "link_code", "mount_hf_model", "workflow_configuration", "forward_environment_variable", "input_docker_software_id", "input_upload_id", "ir_re_ranker", "ir_re_ranking_input", "previous_stages", "tira_image_workdir"]
        for i in to_delete:
            del software_details[i]
        software_details = json.dumps(software_details)
    else:
        software_details = None

    ir_metadata = Path(fn).parent / ".tracking-results.yml"
    if ir_metadata.is_file():
        import yaml
        ir_metadata = yaml.safe_load(ir_metadata.read_text())
        tmp = {}
        for i in ["cpu", "ram"]:
            for j in ["system", "process"]:
                tmp[f"{i}-{j}-max"] = ir_metadata["resources"][i][f"used {j}"]["max"]

        tmp["process-wallclock"] = ir_metadata["resources"]["runtime"]["wallclock"]
        ir_metadata = tmp
    else:
        ir_metadata = {}

    row["software"] = software_details
    if ir_metadata:
        for k, v in ir_metadata.items():
            row[k] = v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        help="directory in which results are stored",
        default="results",
    )
    parser.add_argument(
        "--dataset",
        help="the dataset to evaluate (otherwise look into the .h5 attributes)",
        default=None,
    )
    parser.add_argument(
        "--task",
        help="the task to evaluate (otherwise look into the .h5 attributes)",
        default=None,
    )
    parser.add_argument(
        "--cache",
        help="cache file for faster ",
        default=None,
    )
    parser.add_argument("csvfile")
    args = parser.parse_args()

    gt_cache = {}  # (dataset, task) -> gt_I array
    columns = ["dataset", "task", "algo", "buildtime", "querytime", "params", "recall"]
    row_cache = {}

    if args.cache and Path(args.cache).is_file():
        with open(args.cache, "r") as f:
            for l in f:
                try:
                    l = json.loads(l)
                    row_cache[l["fn"]] = l["row"]
                except:
                    pass
        print(f"Have {len(row_cache)} lines in cache {args.cache}.")

    with open(args.csvfile, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for fn in tqdm(list(get_all_results(args.results))):
            if fn in row_cache:
                writer.writerow(row_cache[fn])
                continue
            attrs, knns = load_results(fn)
            dataset = attrs["dataset"] if not args.dataset else args.dataset
            task = attrs["task"] if not args.task else args.task

            if dataset not in DATASETS or task not in DATASETS[dataset]:
                print(f"Skipping {fn}: unknown dataset={dataset!r} task={task!r}")
                continue

            prepare(dataset, task)

            cache_key = (dataset, task)
            if cache_key not in gt_cache:
                gt_fn = get_fn(dataset, task)
                print(f"Loading ground truth from {gt_fn}")
                with h5py.File(gt_fn) as gf:
                    gt_I = np.array(get_h5_item(gf, DATASETS[dataset][task]["gt_I"]))
                gt_cache[cache_key] = gt_I

            gt_I = gt_cache[cache_key]
            k = DATASETS[dataset][task]["k"]
            recall = get_recall(knns, gt_I, k)
            row = dict(attrs)
            row["recall"] = recall
            add_details_from_tira(fn, row)

            writer.writerow(row)
            
            if args.cache:
                with open(args.cache, "a+") as f:
                    f.write(json.dumps({"fn": fn, "row": row}) + "\n")

