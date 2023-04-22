# RL_18

# Implementing DQN Agent in PyRace Environment and Suggested Improvements

This project involves implementing a Deep Q-Network (DQN) agent in the PyRace environment and exploring possible improvements to the current model.

1. DQN Agent Implementation in PyRace Environment
The DQN agent is integrated into the PyRace environment to enable the agent to learn and perform actions in the racing game. The agent uses a neural network to estimate the Q-values of state-action pairs and selects the action with the highest predicted Q-value.

2. Suggested Improvements for the Current Model
To improve the current model, consider the following modifications:

2.1 **Continuous Radar Values**: The observe function in the PyRace2D class has been updated to provide continuous radar values (in pixels) rather than discrete intervals of 20 pixels. This allows for more precise distance measurements, improving the model's ability to make better driving decisions.

<pre>
```python
def observe(self):
    radars = self.car.radars
    ret = [r[1] for r in radars]
    return ret
```
</pre>

