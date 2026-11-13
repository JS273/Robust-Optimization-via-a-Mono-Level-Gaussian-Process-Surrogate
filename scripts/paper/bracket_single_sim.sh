source scripts/local_env.sh
output_folder="_experiments/RESS/bracket_paper"

"$PYTHON_BIN" run/mono_level_ro.py \
    --config config/paper/bracket.yaml \
    --output_folder $output_folder \
    --seed 123 \
    --tag "bracket"