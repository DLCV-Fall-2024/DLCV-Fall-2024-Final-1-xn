python3 llama_eval_local.py \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --prediction "rag_results/submission.json" \
    --data_root "data" \
    --split "test" \
    --batch_size 1 \
    --temperature 0.1 \
    --verbose