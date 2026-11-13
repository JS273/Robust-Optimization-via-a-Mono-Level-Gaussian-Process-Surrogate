source scripts/local_env.sh
output_folder="_experiments/RESS/scalability_v3"
n_steps=30

# --- PhD Demo
max_samples=10000

"$PYTHON_BIN" run/scalibility_test.py \
     --output_folder $output_folder \
     --sampler classical \
     --J_device cuda \
     --dtype float \
     --n_steps $n_steps \
     --max_samples $max_samples \
     --fix_gp_samples 5000 \
     --fix_J_samples 5000 

"$PYTHON_BIN" run/scalibility_test.py \
    --output_folder $output_folder \
    --sampler matheron \
    --J_device cuda \
    --dtype float \
    --n_steps $n_steps \
    --max_samples $max_samples \
    --fix_gp_samples 5000 \
    --fix_J_samples 5000 