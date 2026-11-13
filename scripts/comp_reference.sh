source scripts/local_env.sh

# Sabater 1d
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater1d --measure quantile --tau 0.95 --model_kwargs '{"design_uq": false}' --n_points 50
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater1d --measure mean_std --alpha 0.0 --model_kwargs '{"design_uq": false}' --n_points 50
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater1d --measure mean_std --alpha 0.5 --model_kwargs '{"design_uq": false}' --n_points 50
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater1d --measure mean_std --alpha 1.0 --model_kwargs '{"design_uq": false}' --n_points 50

# Sabater 2d
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater2d --tau 0.95 --model_kwargs '{"uq_domain": "linear"}' --n_points 20
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater2d --tau 0.95 --model_kwargs '{"uq_domain": "quad"}' --n_points 20
"$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater2d --measure quantile --tau 0.99 --model_kwargs '{"uq_domain": "quad"}' --n_points 20
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater2d --measure mean_std --alpha 0.0 --model_kwargs '{"uq_domain": "quad"}' --n_points 20
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater2d --measure mean_std --alpha 0.5 --model_kwargs '{"uq_domain": "quad"}' --n_points 20
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater2d --measure mean_std --alpha 1.0 --model_kwargs '{"uq_domain": "quad"}' --n_points 20
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Sabater2d --tau 0.95 --model_kwargs '{"uq_domain": "nonlin"}' --n_points 20

# Branin Hoo
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model BraninHooUQ --tau 0.95 --model_kwargs '{"uq_domain": "nonlin"}' --n_points 20

# Hartmann
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Hartmann6 --tau 0.95 --model_kwargs '{"uq_domain": "uq"}' --n_points 50
# "$PYTHON_BIN" run/brute_force_reference.py --output_folder _reference_comp --model Hartmann6 --tau 0.8 --model_kwargs '{"uq_domain": "uq"}' --n_points 50

# Bracket
# "$PYTHON_BIN" run/evo_sbuq_ro.py --config scripts/config/bracket_ref.yaml

