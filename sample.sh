#!/bin/bash

# This bash script runs the proposed llm semantic gp with the required parameters

# WARNING: Do not forget to run this in a python venv environment with all the
#          needed dependencies

problems=()
runs=()

# Add any number of problems from LoadDataset.py
for i in {27..27}
do
    problems+=($i)
done
# for i in {5..5}            # Problem number 4 to 5
# do
#     problems+=($i)
# done

for prob in "${problems[@]}"
do
    echo "$prob"
done

# Outter loop is the individual run (i.e., we increment the seed after each problem)
# has been run at least once
for j in {6..30}
do
    # bla=$((j + 42))
    # Start the LLM model worker with the new seed here (This won't work as the script
    # will hang here! Need to daemonize this process, and find a way to kill it later)
    # We need to dispatch jobs here and kill them
    tmux new-session -d -s model_worker_1 "source /home/someone/documents/llmgpsr/bin/activate;
                                        CUDA_VISIBLE_DEVICES=1 \
                                        python3 -m fastchat.serve.model_worker --gpus 1 --num-gpus 1 --seed $j \
                                        --model-path /home/someone/documents/llm_models --port 31000 \
                                        --worker http://localhost:31000 \
                                        --controller http://localhost:21001"

    tmux new-session -d -s model_worker_2 "source /home/someone/documents/llmgpsr/bin/activate;
                                        CUDA_VISIBLE_DEVICES=2 \
                                        python3 -m fastchat.serve.model_worker --gpus 2 --num-gpus 1 --seed $j \
                                        --model-path /home/someone/documents/llm_models --port 31001 \
                                        --worker http://localhost:31001 \
                                        --controller http://localhost:21001"

    # Wait for the model worker to properly start
    sleep 120

    python3 ./sample.py "${problems[@]}" $j

    # End tmux model_worker session for run with new seed
    tmux kill-session -t model_worker_1
    tmux kill-session -t model_worker_2

done
