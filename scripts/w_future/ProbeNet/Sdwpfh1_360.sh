python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "Sdwpfh1.csv" \
    --strategy-args '{"horizon": 360, "target_channel": [-1]}' \
    --model-name "probenet.ProbeNet" \
    --model-hyper-params '{
        "batch_size": 64,
        "dropout": 0.0,
        "horizon": 360,
        "loss": "MAE",
        "lr": 0.0001,
        "lradj": "type3",
        "norm": true,
        "num_epochs": 50,
        "patience": 5,
        "seq_len": 720,
        "alpha": 0.9,
        "baseline": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 24
        },
        "probe": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 24,
            "n_templates": 4,
            "beta": 0.01
        }
    }' \
    --gpus 6 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "Sdwpfh1/ProbeNet"
