source scripts/local_env.sh
output_folder="_experiments/RESS/hartmann_batch_run"

for seed in {1..10}; do
    "$PYTHON_BIN" run/mono_level_ro.py \
        --config config/paper/hartmann6.yaml \
        --n_doe 90 \
        --n_EI 108 \
        --n_EM 36 \
        --n_TR 36 \
        --model_kwargs '{"uq_domain": "uq"}' \
        --output_folder $output_folder \
        --seed "$seed" \
        --tag "ro_$seed"
done
