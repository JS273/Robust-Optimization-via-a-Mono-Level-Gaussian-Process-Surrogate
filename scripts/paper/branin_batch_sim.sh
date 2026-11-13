source scripts/local_env.sh
output_folder="_experiments/RESS/branin_batch_run_paper"

for seed in {1..10}; do
    "$PYTHON_BIN" run/mono_level_ro.py \
        --config config/paper/braninhoo.yaml \
        --model_kwargs '{"uq_domain": "nonlin"}' \
        --output_folder $output_folder \
        --seed "$seed" \
        --tag "nonlin_uq$seed"
done


