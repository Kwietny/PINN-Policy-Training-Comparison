# PINN-Policy-Training-Comparison
## Abstract
This project serves as a demo of using a Physics Inspired Neural Network as a subsitute of a real environment for purposes of Reinforcement Learning (in this case DQN). 
The idea is to approximate a real environment which is very expensive to simulate(for example a real world) so that we can train our RL model relatively cheaply.
Although we use simple lunar lander simulation(which is actually not very expensive to run) in theory this approach can be transferred to any RL problem.

## Sources
- https://gymnasium.farama.org/environments/box2d/lunar_lander/ - our simulation framework
- https://github.com/wtcherr/lunar-lander-dqn - baseline solution 
