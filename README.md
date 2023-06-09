# RL_18

# Implementing DQN Agent in PyRace Environment and Suggested Improvements

This project involves implementing a Deep Q-Network (DQN) agent in the PyRace environment and exploring possible improvements to the current model.

## 1. DQN Agent Implementation in PyRace Environment
The DQN agent is integrated into the PyRace environment to enable the agent to learn and perform actions in the racing game. The agent uses a neural network to estimate the Q-values of state-action pairs and selects the action with the highest predicted Q-value.

## 2. Suggested/Applied Improvements for the Current Model

## **Suggested Improvements**:




- Construct a better model architecture:
    Experimenting with more complex model architectures, such as adding more layers or incorporating different layer types (e.g., convolutional layers, recurrent layers). We could give as an input the images of the game to the agent and use the cvl layers to get better features as an example.The use of recurrent layers couls also help taking in account the sequence of the race , thus better anticipationg turns and future actions to make. 
- Explore and experimenting more RL algorithms:
    Implementing advanced reinforcement learning algorithms like Double DQN, Dueling DQN, Proximal Policy Optimization (PPO), or Deep Deterministic Policy Gradient (DDPG) for continuous action spaces.
- Hyperparameter tuning for the model:
    Tuning hyperparameters, such as learning rates, discount factors, exploration rates, and memory buffer sizes.
    Exploring better exploration strategies, such as Upper Confidence Bound (UCB), or Thompson Sampling.
- Find a more efficient exploration strategy:
    Instead of using an epsilon-greedy strategy, you can experiment with other exploration strategies, such as Upper Confidence Bound (UCB), or Thompson Sampling that we have covered during the course.

## **Applied Modifications**:


## 2.1 **Continuous Radar Values**:
The observe function in the PyRace2D class has been updated to provide continuous radar values (in pixels) rather than discrete intervals of 20 pixels. This allows for more precise distance measurements, improving the model's ability to make better driving decisions.

<pre>
``` python

def observe(self):
    radars = self.car.radars
    ret = [r[1] for r in radars]
    return ret
```
</pre>

## 2.2 **Continuous Actions**

**Continuous Action Space**: The action space has been changed from a discrete set of actions to a continuous action space using gym.spaces.Box. This allows for more precise control of acceleration, steering angle, and braking.

<pre>
``` python
# In the RaceEnv class
self.action_space = spaces.Box(low=np.array([-1.0, -1.0, 0.0]),
                               high=np.array([1.0, 1.0, 1.0]),
                               dtype=np.float32)
```
</pre>

**Continuous Action Handling**: The action function in the Car class has been updated to handle continuous actions for acceleration, steering angle, and braking. The input action is now an array of three values, where the first value represents acceleration, the second value represents steering angle change, and the third value represents braking.

### 2.3 **Adjusted Friction**: The update function in the Car class has been modified to use a proportional friction rate, making the deceleration more realistic.

<pre>
``` python
def update(self, map=None):
    friction = 0.05 * self.speed  # Adjust the 0.05 value to change friction

    self.speed -= friction
    if self.speed > 10:
        self.speed = 10
    if self.speed < 1:
        self.speed = 1
```
</pre>

To use the modified Gym Continuous PyRace2D environment, clone this repository and run the Py folder with the new Pyrace_RL_DQNMODEL_Continious.PY script, it uses the modified version of gym_race which is pyrace_gym_continious. 
To run this repository install the dependencies

## **Optional More Advanced Model: Pyrace_RL_A2C_Continious**
`Pyrace_RL_A2C_Continious` uses the A2C (Advantage Actor-Critic) algorithm to train an agent to play a continuous racing game in the PyRace environment. The agent interacts with the environment, stores the experience tuples in the memory, and updates the model using the A2C algorithm. The A2C model class defines the actor-critic network with three dense layers. During training, the agent uses the A2C algorithm to update the model by computing advantages and gradients. The learning rate and exploration rate are annealed over time. The `simulate` function runs the game loop and evaluates the model after each episode. 


To use the modified Gym Continuous PyRace2D environment, clone this repository and run the Py folder with the new python3 `Pyrace_RL_A2C_Continious.py` script, it uses the modified version of gym_race which is `pyrace_gym_continious`(the same as the one before). 



