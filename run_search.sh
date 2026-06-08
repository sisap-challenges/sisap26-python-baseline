run_for_name() {
    local name="$1"
    echo "Running for dataset: $name"
    mkdir -p "results/$name"
    docker run \
        --rm \
        --user "$(id -u):$(id -g)" \
        --cpus=4 \
        --memory=16g \
        --memory-swap=16g \
        --memory-swappiness 0 \
        --volume $(pwd)/search.py:/app/search.py:ro \
        --volume $(pwd)/data:/app/data:ro \
        --volume $(pwd)/results:/app/results:rw \
        sisap-baseline python search.py \
            --input "data/$name/"*.h5 \
            --task-description "data/$name/config.json" \
            --output "results/$name/"
}

if [ $# -eq 0 ]; then
    for task in 1 2 3; do
        run_for_name "task-$task-spot-check"
    done
else
    run_for_name "$1"
fi
