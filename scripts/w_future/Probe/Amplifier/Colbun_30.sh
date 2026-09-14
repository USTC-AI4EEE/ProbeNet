python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "Colbun.csv" \
    --strategy-args '{"horizon": 30, "target_channel": [-1]}' \
    --model-name "amplifier.AmplifierProbe" \
    --model-hyper-params '{
        "batch_size": 32,
        "horizon": 30,
        "lr": 0.001,
        "norm": true,
        "seq_len": 180,
        "alpha": 0.1,
        "baseline": {
            "SCI": 0,
            "hidden_size": 256
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
    --gpus 1 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "Colbun/AmplifierProbe"
