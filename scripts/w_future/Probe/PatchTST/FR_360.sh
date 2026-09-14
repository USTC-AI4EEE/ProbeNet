python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "FR.csv" \
    --strategy-args '{"horizon": 360, "target_channel": [-1]}' \
    --model-name "time_series_library.PatchTSTProbe" \
    --model-hyper-params '{
        "horizon": 360,
        "lr": 0.0001,
        "norm": true,
        "seq_len": 720,
        "alpha": 0.3,
        "baseline": {
            "d_ff": 1024,
            "d_model": 128,
            "e_layers": 1,
            "factor": 3,
            "n_heads": 2
        },
        "probe": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 24,
            "n_templates": 2,
            "beta": 0.1,
            "dropout": 0.0
        }
    }' \
    --gpus 5 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "FR/PatchTSTProbe"
