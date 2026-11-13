source scripts/local_env.sh
output_folder="_experiments/RESS/BraninHoo/transfer_learning_batch_alpha0p5"

# Reference sim
for seed in {1..10}; do
    "$PYTHON_BIN" run/mono_level_ro.py \
        --config config/paper/braninhoo.yaml \
        --measure mean_std \
        --alpha 0.5 \
        --n_doe_se 0 \
        --n_doe_ufun 0 \
        --output_folder $output_folder \
        --seed "$seed" \
        --tag "ref_mean_std_0p5_seed$seed"
done

# Reuse sim
for seed in {1..10}; do
    run_seed=$((seed + 0))
    "$PYTHON_BIN" run/mono_level_ro.py \
        --config config/paper/braninhoo.yaml \
        --sim_file "_experiments/RESS/branin_batch_run_paper/nonlin_uq${seed}_data.pkl"\
        --transfer_learning \
        --load_EI \
        --conv_EI '{"switch": 1, "eps": 0.005, "n_ave": 3}' \
        --measure mean_std \
        --alpha 0.5 \
        --n_doe_se 0 \
        --n_doe_ufun 0 \
        --output_folder $output_folder \
        --seed "$run_seed" \
        --tag "reuse_mean_std_0p5_seed$run_seed"
done