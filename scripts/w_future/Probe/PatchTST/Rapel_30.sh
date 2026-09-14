python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "Rapel.csv" \
    --strategy-args '{"horizon": 30, "target_channel": [-1]}' \
    --model-name "time_series_library.PatchTSTProbe" \
    --model-hyper-params '{
        "horizon": 30,
        "lr": 0.001,
        "norm": true,
        "seq_len": 180,
        "alpha": 0.1,
        "baseline": {
            "d_ff": 2048,
            "d_model": 128,
            "e_layers": 1,
            "factor": 3,
            "n_heads": 2
        },
        "probe": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 5,
            "n_templates": 4,
            "beta": 0.5,
            "dropout": 0.0
        }
    }' \
    --gpus 6 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "Rapel/PatchTSTProbe"
