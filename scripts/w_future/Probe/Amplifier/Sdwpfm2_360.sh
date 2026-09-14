python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "Sdwpfm2.csv" \
    --strategy-args '{"horizon": 360, "target_channel": [-1]}' \
    --model-name "amplifier.AmplifierProbe" \
    --model-hyper-params '{
        "batch_size": 32,
        "horizon": 360,
        "lr": 0.001,
        "norm": true,
        "seq_len": 720,
        "alpha": 0.7,
        "baseline": {
            "SCI": 1,
            "hidden_size": 256
        },
        "probe": {
            "d_ff": 128,
            "d_model": 64,
            "n_heads": 4,
            "patch_len": 24,
            "n_templates": 4,
            "beta": 0.1
        }
    }' \
    --gpus 7 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "Sdwpfm2/AmplifierProbe"
