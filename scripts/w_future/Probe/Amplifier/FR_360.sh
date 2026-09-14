python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "FR.csv" \
    --strategy-args '{"horizon": 360, "target_channel": [-1]}' \
    --model-name "amplifier.AmplifierProbe" \
    --model-hyper-params '{
        "batch_size": 32,
        "horizon": 360,
        "lr": 0.0005,
        "norm": true,
        "seq_len": 720,
        "alpha": 0.3,
        "baseline": {
            "SCI": 0,
            "hidden_size": 32
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
    --gpus 5 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "FR/AmplifierProbe"
