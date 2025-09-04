# Coding Trajectory Collection for AgenTracer

## Prepare Coding Datasets
the scripts for preparing coding datasets are in `prepare_datasets` folder.
prepare scripts do following things:
- download the dataset from the source
- filter out the tasks with broken tests
- save the tasks into `parquet` format

Columns for each dataset:
- `data_source`: the name of the dataset
- `task_id`: the id of the task
- `question`: the question of the task
- `test`: the test code of the task
- `entry_point`: the entry point of the task
- `reference_solution`: the reference solution of the task

```bash
# KodCode/KodCode-V1
python prepare_datasets/kodcode.py
    --save_dir datasets/code/  # by default, save to `datasets/code/`
    --difficulty \["hard", "medium", "easy"\] # by default, hard only
    
# [WIP] More
```

## Sandbox-Fusion Local Deployment for Evaluation
start a sandbox-fusion server using docker.
according to [sandbox-fusion docs](https://bytedance.github.io/SandboxFusion/docs/docs/get-started), you can start a server with:
```bash
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```
Edit `utils.py` to set the endpoint to the server(localhost:8080), now using ByteDance's FaSS server.

## Run Rollout
```bash
python run_rollout.py
    --dataset datasets/code/kodcode-light-rl-10k-hard.parquet
    --concurrency 10
    --skip_existing
```

## Run Evaluation
results will be saved in `eval_results` folder, named as `{data_source}.json`
```bash
python run_eval.py
    --dataset datasets/code/kodcode-light-rl-10k-hard.parquet
    --concurrency 10
```