python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "Sdwpfm1.csv" \
    --strategy-args '{"horizon": 360, "target_channel": [-1]}' \
    --model-name "timekan.TimeKANProbe" \
    --model-hyper-params '{
        "batch_size": 64,
        "horizon": 360,
        "lr": 0.0001,
        "norm": true,
        "num_epochs": 10,
        "patience": 3,
        "seq_len": 720,
        "alpha": 0.7,
        "baseline": {
            "begin_order": 0,
            "d_ff": 16,
            "d_model": 16,
            "down_sampling_layer": 1,
            "down_sampling_window": 2,
            "e_layers": 2
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
    --gpus 7 \
    --num-workers 1 \
    --timeout 60000 \
    --deterministic full \
    --save-path "Sdwpfm1/TimeKANProbe"
