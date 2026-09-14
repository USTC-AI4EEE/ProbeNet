python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "BE.csv" \
    --strategy-args '{"horizon": 24, "target_channel": [-1]}' \
    --model-name "probenet.ProbeNet" \
    --model-hyper-params '{
        "batch_size": 64,
        "dropout": 0.0,
        "horizon": 24,
        "loss": "MAE",
        "lr": 0.001,
        "lradj": "type3",
        "norm": true,
        "num_epochs": 50,
        "patience": 5,
        "seq_len": 168,
        "alpha": 0.3,
        "baseline": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 36
        },
        "probe": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 24,
            "n_templates": 2,
            "beta": 0.1
        }
    }' \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "BE/ProbeNet"
