# DarkBotX

Copyright (c) 2026 Avik Md Emtiaz Arefin

## GETTING STARTED
Git clone the project then run the command below

```
./setup.sh
```

## Train 
```
uv run src/train.py 
```

## Eval
```
uv run src/eval.py
```

## Troubleshoot
Incase the setup fails, you can manually install uv on your system then run the script below to get started.

```
uv init --python 3.12
uv add torch genesis-world rsl-rl-lib pygame tensorboard mtrick
```


## Run mtrick

```
uv run mtrick
```

And a link will pop up, ctrl / cmd click on the link.

# Setup Pi

Might need to use something else. This was a WIP option.
```bash
sudo apt install liblgpio-dev
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project utilizes [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) as its primary physics simulator.

```bibtex
@article{
   genesis2026genesisworld,
   author = {Genesis AI Team},
   title = {The Role of Simulation in Scalable Robotics, Genesis World 1.0, and the Path Forward},
   journal = {Genesis AI Blog},
   month = {May},
   year = {2026},
   url = {https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward},
}
@misc{
  Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Generative and Universal Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/genesis-world}
}
```

## TODO
2. Add an angle‑centric auxiliary reward
3. Penalise unnecessary rotation after lift