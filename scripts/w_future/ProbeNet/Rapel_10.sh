python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "Rapel.csv" \
    --strategy-args '{"horizon": 10, "target_channel": [-1]}' \
    --model-name "probenet.ProbeNet" \
    --model-hyper-params '{
        "batch_size": 64,
        "dropout": 0.0,
        "horizon": 10,
        "loss": "MAE",
        "lr": 0.001,
        "lradj": "type3",
        "norm": true,
        "num_epochs": 50,
        "patience": 5,
        "seq_len": 60,
        "alpha": 0.1,
        "baseline": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 10
        },
        "probe": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 5,
            "n_templates": 4,
            "beta": 0.5
        }
    }' \
    --gpus 3 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "Rapel/ProbeNet"
