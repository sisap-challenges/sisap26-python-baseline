import argparse
import faiss
import h5py
import numpy as np
import os
from pathlib import Path
import time
from tqdm import tqdm
from scipy.sparse import csr_matrix
from datasets import DATASETS, prepare, get_fn

def store_results(dst, algo, dataset, task, D, I, buildtime, querytime, params):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['dataset'] = dataset
    f.attrs['task'] = task
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()

def run_task2(dataset, task, k):
    print(f'Running {task} on {dataset}')

    prepare(dataset, task)

    fn = get_fn(dataset, task)
    f = h5py.File(fn)
    data = np.array(DATASETS[dataset][task]['data'](f))
    queries = np.array(DATASETS[dataset][task]['queries'](f))
    f.close()

    n, d = data.shape
    if task == 'task2':
        k = k + 1 # need to search for one more NN since we cannot remove self-loop

    nlist = 1024 # number of clusters/centroids to build the IVF from
    index_identifier = f"IVF{nlist},SQfp16"

    index = faiss.index_factory(d, index_identifier, faiss.METRIC_INNER_PRODUCT)

    print(f"Training index on {data.shape}")
    start = time.time()
    index.train(data)
    index.add(data)
    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")
    assert index.is_trained

    for nprobe in [1, 2, 5, 10, 100]:
        print(f"Starting search on {queries.shape} with nprobe={nprobe}")
        start = time.time()
        index.nprobe = nprobe
        D, I = index.search(queries, k)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        I = I + 1 # FAISS is 0-indexed, groundtruth is 1-indexed

        identifier = f"index=({index_identifier}),query=(nprobe={nprobe})"

        store_results(os.path.join("results/", dataset, task, f"{identifier}.h5"), "faissIVF", 
                      dataset, task, D, I, elapsed_build, elapsed_search, identifier)

def run_task3(dataset, task, k):
    print(f'Running {task} on {dataset}')

    prepare(dataset, task)

    fn = get_fn(dataset, task)
    f = h5py.File(fn)
    corpus = DATASETS[dataset][task]['data'](f)
    queries = DATASETS[dataset][task]['queries'](f)
    f.close()

    print(f"Corpus shape: {corpus.shape}")
    print(f"Queries shape: {queries.shape}")

    n_queries = queries.shape[0]

    I = np.zeros((n_queries, k), dtype=np.int32)
    D = np.zeros((n_queries, k), dtype=np.float32)

    start_time = time.time()
    print(f"Extracting top-{k} neighbors...")
    # Using scipy sparse exact search query by query
    for i in tqdm(range(n_queries), desc="Processing Queries"):
        q_dense = queries[i].toarray().flatten()
        row_scores = corpus.dot(q_dense)
        
        top_k_idx = np.argpartition(-row_scores, k)[:k]
        top_k_sorted_idx = top_k_idx[np.argsort(-row_scores[top_k_idx])]
        
        I[i] = top_k_sorted_idx + 1  # 1-based indexing for GT matches
        D[i] = row_scores[top_k_sorted_idx]
            
    elapsed_search = time.time() - start_time
    print(f"Extraction completed in {elapsed_search:.2f} seconds.")

    identifier = "scipy_exact"
    store_results(os.path.join("results/", dataset, task, f"{identifier}.h5"), identifier, 
                  dataset, task, D, I, 0.0, elapsed_search, identifier)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=['task1', 'task2', 'task3'],
        default='task2'
    )

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='llama-dev'
    )


    args = parser.parse_args()
    if args.task == 'task3':
        run_task3(args.dataset, args.task, DATASETS[args.dataset][args.task]['k'])
    else:
        run_task2(args.dataset, args.task, DATASETS[args.dataset][args.task]['k'])

