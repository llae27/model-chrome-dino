# model-chrome-dino

## Installation

The chrome driver is set to match chrome major version 145.

### Linux chrome

If you're working on linux, use the following command to download full chrome.

`wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb`

### `gym-chrome-dino` updated

Use the following line to install gym library in your environment.

`pip install -e ./bin/gym-chrome-dino`

If you can't load the library properly, initialize and update submodules first.

```
git submodule init
git submodule update --remote --recursive
```

## Gymnasium usage

The original documentation includes few bugs. I've listed fixes below.

To turn on the default acceleration of the game, add the following line.

`env.unwrapped.set_acceleration(True)`

To set a custom value for the acceleration, use the following line.

`env.unwrapped.game.set_parameter("config.ACCELERATION", [VALUE])`