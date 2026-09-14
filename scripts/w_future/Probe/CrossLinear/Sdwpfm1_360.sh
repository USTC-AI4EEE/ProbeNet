python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "Sdwpfm1.csv" \
    --strategy-args '{"horizon": 360, "target_channel": [-1]}' \
    --model-name "crosslinear.CrossLinearProbe" \
    --model-hyper-params '{
        "batch_size": 16,
        "horizon": 360,
        "loss": "MSE",
        "lr": 0.001,
        "lradj": "type1",
        "norm": true,
        "num_epochs": 10,
        "patience": 3,
        "seq_len": 720,
        "alpha": 0.7,
        "baseline": {
            "alpha": 0,
            "beta": 0.5,
            "d_ff": 1024,
            "d_model": 256,
            "patch_len": 8
        },
        "probe": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 24,
            "n_templates": 4,
            "beta": 0.1,
            "dropout": 0.0
        }
    }' \
    --gpus 5 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "Sdwpfm1/CrossLinearProbe"
