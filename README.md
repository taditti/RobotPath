# RobotPath

![Alt Text](https://github.com/kavehkamali/RobotPath/blob/master/demo.gif)


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

Note 1: OpenAI baselines will install gym which need MuJuco licence. We do not need MuJuco so just remove the words: "mujoco,robotics" from "setup.py" before running the above command.

Note 2: Atary module of gym cannot be installed on windows. So remove the word: "atari" from "setup.py" befor running it.

3- Install MPI on your system and put its path in the PATH environmetal variable.

4- Run RobotPath training:

```
python main.py
```