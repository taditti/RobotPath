# RobotPath

## Using OpenAI DDPG for online obstacle avoidance.

<img src="https://github.com/kavehkamali/RobotPath/blob/master/demo.gif" width="200">
![](https://github.com/kavehkamali/RobotPath/blob/master/demo.gif = 250x250)


## Installation:

1- Install pybullet:

```
pip install pybullet
```

2- Install OpenAI baselines:

```
https://github.com/openai/baselines
```
For installing baselines you need to run:

```
pip install -e .
```

Note 1: OpenAI baselines will install gym which needs MuJuco license. We do not need MuJuco so just remove the words: "mujoco,robotics" from "setup.py" before running the above command.

Note 2: Atary module of gym cannot be installed on windows. So remove the word: "Atari" from "setup.py" before running it.

3- Install MPI on your system and put its path in the PATH environmental variable.

## Training:

Run RobotPath training:

```
python main.py
```