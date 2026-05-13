# Invariant Representation Learning for Source-Free Time Series Forecasting with LLM-Centric Proxy Denoising

## Running

- **Data Preparation:** *Weather*, *Traffic*, *Electricity* and *ETT* can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Please put each `.csv` file in `./dataset/`.

- **Training Source Model:** Run scripts like `etth1.sh` in  `./scripts/` to train the source model on the corresponding dataset, for example

  ```bash
  bash ./scripts/etth1.sh
  ```

- **Adaptation to Target domain:** After obtaining the source model, run scripts like `etth1_weather.sh` in `./scripts/` to achieve adaptation to target domain,  for example

  ```
  bash ./scripts/etth1_weather.sh
  ```


## Requirements

```
python==3.8
einops==0.8.1
matplotlib==3.7.5
numpy==1.24.4
pandas==2.0.3
ptflops==0.7.5
reformer_pytorch==1.4.4
scikit_learn==1.3.2
seaborn==0.13.2
torch==2.4.1
torchvision==0.12.0+cu113
tqdm==4.67.1
transformers==4.46.3
tsai==0.4.1
```







