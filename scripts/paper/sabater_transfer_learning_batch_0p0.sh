source scripts/local_env.sh
output_folder="_experiments/RESS/Sabater2d/transfer_learning_batch_alpha0p0"

# Reference sim
for seed in {1..10}; do
    "$PYTHON_BIN" run/mono_level_ro.py \
        --config config/paper/sabater2d.yaml \
        --model_kwargs '{"uq_domain": "quad"}' \
        --measure mean_std \
        --alpha 0.0 \
        --output_folder $output_folder \
        --seed "$seed" \
        --tag "ref_mean_std_0_seed$seed"
done

# Reuse sim
for seed in {1..10}; do
    run_seed=$((seed + 0))
    "$PYTHON_BIN" run/mono_level_ro.py \
        --config config/paper/sabater2d.yaml \
        --model_kwargs '{"uq_domain": "quad"}' \
        --measure mean_std \
        --alpha 0.0 \
        --sim_file "_experiments/RESS/sabater_batch_run_paper/quad_uq${seed}_data.pkl"\
        --transfer_learning \
        --load_EI \
        --conv_EI '{"switch": 1, "eps": 0.005, "n_ave": 3}' \
        --output_folder $output_folder \
        --seed "$run_seed" \
        --tag "reuse_mean_std_0_seed$run_seed"
done