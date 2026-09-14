python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "NP.csv" \
    --strategy-args '{"horizon": 360, "target_channel": [-1]}' \
    --model-name "crosslinear.CrossLinearProbe" \
    --model-hyper-params '{
        "batch_size": 16,
        "horizon": 360,
        "loss": "MSE",
        "lr": 0.0001,
        "lradj": "type1",
        "norm": true,
        "num_epochs": 10,
        "patience": 3,
        "seq_len": 720,
        "alpha": 0.3,
        "baseline": {
            "alpha": 0.5,
            "beta": 1.0,
            "d_ff": 1024,
            "d_model": 256,
            "patch_len": 8
        },
        "probe": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 24,
            "n_templates": 2,
            "dropout": 0.0,
            "beta": 0.1
        }
    }' \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "NP/CrossLinearProbe"
