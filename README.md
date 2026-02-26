# DarkBotX

## GETTING STARTED
Git clone the project then run the command below

```
./setup.sh
```

## Run the project
```
uv run main.py
```


## Troubleshoot
Incase the setup fails, you can manually install uv on your system then run the script below to get started.

```
uv init --python 3.12
uv add torch genesis-world rsl-rl-lib pyqt6 tensorboard wandb
```


## Setup Wandb

Before running the code, ensure you have `uv add wandb` already.

Create an account on wandb and then copy the api, and use then code below and then paste the api
```bash
uv run wandb login --relogin
```

then like regularly

```bash
uv run main.py -t --debug sim
```

or whatever you want