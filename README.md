# ProbeNet: Augmenting Time Series Forecasting with Future-Known Exogenous Variables via Patch-Local Exogenous-to-Target Modeling


## Quickstart

### Prepare Data
You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1K2AvogpOpSz1PiQ53dPchzGv_PqlCWAK/view?usp=sharing). Then place the downloaded data under the folder `./dataset`. 

### Installation
> [!IMPORTANT]
> this project is fully tested under python 3.11, it is recommended that you set the Python version to 3.11.
1. Clone this repository.
   ```bash
   git clone https://github.com/mallocobject/ProbeNet.git
   cd ProbeNet
   ```

2. Create a new Conda environment.
   ```bash
   conda create -n probe python=3.11
   conda activate probe
   ```

3. Install Core Dependencies
   > ⚠️ **CUDA Compatibility Notice**
   > The torch prebuilt package is **CUDA-version specific**. (See https://pytorch.org/get-started/previous-versions/)
   > Please make sure to install the package that matches your local CUDA version (e.g., `cu118` or `cu121`).
   > Recommended: torch==2.5.1

   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```


### Train and Evaluate

- To see the model structure of ProbeNet,  [click here](./ts_benchmark/baselines/probenet/model/probe.py).
- We provide all the experiment scripts for ProbeNet and other baselines under the folder `./scripts/w_future`.  For example you can reproduce all the experiment results as the following script:

```shell
sh ./scripts/w_future/ProbeNet.sh
```