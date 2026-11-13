source scripts/local_env.sh
output_folder="_experiments/RESS/sabater_batch_run_paper"

for seed in {1..10}; do
    "$PYTHON_BIN" run/mono_level_ro.py \
        --config config/paper/sabater2d.yaml \
        --model_kwargs '{"uq_domain": "linear"}' \
        --output_folder $output_folder \
        --seed "$seed" \
        --tag "linear_uq$seed"
done

for seed in {1..10}; do
    "$PYTHON_BIN" run/mono_level_ro.py \
        --config config/paper/sabater2d.yaml \
        --model_kwargs '{"uq_domain": "quad"}' \
        --output_folder $output_folder \
        --seed "$seed" \
        --tag "quad_uq$seed"
done

for seed in {1..10}; do
    "$PYTHON_BIN" run/mono_level_ro.py \
        --config config/paper/sabater2d.yaml \
        --model_kwargs '{"uq_domain": "nonlin"}' \
        --output_folder $output_folder \
        --seed "$seed" \
        --tag "nonlin_uq$seed"
done


