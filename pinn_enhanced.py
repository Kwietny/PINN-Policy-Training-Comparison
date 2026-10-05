import random
import datetime
import gymnasium as gym
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
from dqn import DQN
from reward_model import RewardModel, RewardTrainer

from dynamics import PhysicsResidualDynamics, DynamicsTrainer, STATE_NAMES, evaluate_dynamics

# Initialize TensorBoard writer
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f'runs/pinn_res_rewest_1000_50_10_{current_time}')

num_episodes = 300
batch_size = 128
GAMMA = 0.99
LR = 1e-4
TAU = 0.005

EPSILON = 1.0  # Start with full exploration
EPSILON_MIN = 0.01  # Minimum value
EPSILON_DECAY = 0.995  # Decay factor per episode

REPLAY_CAPACITY = 10000

# Dynamics model
DYNAMICS_UPDATE_INTERVAL = 25
DYNAMICS_EPOCHS = 20
DYNAMICS_BATCH_SIZE = 256

# How much model-generated experience is used
MODEL_BATCH_SIZE = 64
REAL_BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple(
    "Transition", ["state", "action", "next_state", "reward", "done"]
)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(
            self.memory,
            batch_size
        )

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


# Initialize the environment
env = gym.make("LunarLander-v3") # , render_mode="human"

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

real_memory = ReplayMemory(REPLAY_CAPACITY)

model_memory = ReplayMemory(REPLAY_CAPACITY)

dynamics_model = PhysicsResidualDynamics(
    dt=1.0
).to(device)

dynamics_trainer = DynamicsTrainer(
    dynamics_model,
    device=device,
    learning_rate=1e-3,
    physics_weight=0.0
)

reward_model = RewardModel().to(device)

reward_trainer = RewardTrainer(
    reward_model,
    device=device,
    learning_rate=1e-3
)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()


def select_action(state):
    if np.random.rand() < EPSILON:
        return torch.tensor(
            [[env.action_space.sample()]], dtype=torch.long, device=device
        )
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)  # Exploit (best action)

def optimize_dqn():
    total_available = (
        len(real_memory)
        + len(model_memory)
    )

    if total_available < batch_size:
        return None

    # ----------------------------------
    # Prefer a balanced batch
    # ----------------------------------

    real_count = min(
        REAL_BATCH_SIZE,
        len(real_memory)
    )

    model_count = min(
        MODEL_BATCH_SIZE,
        len(model_memory)
    )

    # If model memory isn't available yet,
    # use entirely real data.
    if model_count == 0:
        real_count = batch_size
        model_count = 0

    # Fill remaining slots from real data
    if real_count + model_count < batch_size:

        additional = (
            batch_size
            - real_count
            - model_count
        )

        real_count = min(
            len(real_memory),
            real_count + additional
        )

    transitions = []

    if real_count > 0:
        transitions.extend(
            random.sample(
                real_memory.memory,
                real_count
            )
        )

    if model_count > 0:
        transitions.extend(
            random.sample(
                model_memory.memory,
                model_count
            )
        )

    states, actions, next_states, rewards, dones = zip(
        *transitions
    )

    states_batch = torch.cat(states)
    actions_batch = torch.cat(actions)
    next_states_batch = torch.cat(next_states)

    rewards_batch = torch.cat(rewards)

    dones_batch = torch.tensor(
        dones,
        dtype=torch.bool,
        device=device
    )

    with torch.no_grad():

        next_q = (
            target_net(next_states_batch)
            .max(1)[0]
        )

        q_target = (
            rewards_batch
            + GAMMA
            * next_q
            * (~dones_batch)
        )

    q_policy = (
        policy_net(states_batch)
        .gather(
            1,
            actions_batch
        )
    )

    loss = criterion(
        q_policy,
        q_target.unsqueeze(1)
    )

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_value_(
        policy_net.parameters(),
        100
    )

    optimizer.step()

    return loss.item()


def train_dynamics():
    if len(real_memory) < DYNAMICS_BATCH_SIZE:
        return None

    last_result = None

    for _ in range(DYNAMICS_EPOCHS):

        transitions = random.sample(
            real_memory.memory,
            DYNAMICS_BATCH_SIZE
        )

        states, actions, next_states, _, _ = zip(
            *transitions
        )

        states_batch = torch.cat(states)
        actions_batch = torch.cat(actions)
        next_states_batch = torch.cat(next_states)

        last_result = dynamics_trainer.train_batch(
            states_batch,
            actions_batch,
            next_states_batch
        )

    return last_result

def train_reward_model():

    if len(real_memory) < DYNAMICS_BATCH_SIZE:
        return None

    last_loss = None

    for _ in range(DYNAMICS_EPOCHS):

        transitions = random.sample(
            real_memory.memory,
            DYNAMICS_BATCH_SIZE
        )

        states, actions, next_states, rewards, _ = zip(
            *transitions
        )

        states = torch.cat(states)
        actions = torch.cat(actions)
        next_states = torch.cat(next_states)
        rewards = torch.cat(rewards)

        last_loss = reward_trainer.train_batch(
            states,
            actions,
            next_states,
            rewards
        )

    return last_loss


def generate_model_transitions(
    num_samples=1000
):
    if len(real_memory) == 0:
        return 0

    dynamics_model.eval()

    samples = random.choices(
        real_memory.memory,
        k=num_samples
    )

    states = torch.cat(
        [sample.state for sample in samples]
    )

    actions = torch.cat(
        [sample.action for sample in samples]
    )

    with torch.no_grad():

        predicted_next_states = dynamics_model(
            states,
            actions
        )

    for i in range(num_samples):

        state = states[i:i+1]
        action = actions[i:i+1]

        next_state = (
            predicted_next_states[i:i+1]
            .clamp(-10.0, 10.0)
        )

        # reward approximation
        with torch.no_grad():
            predicted_next_state = dynamics_model(
                state,
                action
            )

            predicted_reward = reward_model(
                state,
                action,
                predicted_next_state
            )

            reward = predicted_reward.detach().view(1)

        # done = estimate_done(
        #     next_state
        # )
        done = samples[i].done

        model_memory.push(
            state.detach(),
            action.detach(),
            next_state.detach(),
            reward,
            done
        )

    dynamics_model.train()

    return num_samples

def estimate_reward(
    state,
    action,
    next_state
):

    # Position relative to landing region
    x = next_state[0, 0]
    y = next_state[0, 1]

    vx = next_state[0, 2]
    vy = next_state[0, 3]

    angle = next_state[0, 4]

    reward = 0.0

    # Stay near center
    reward -= 0.5 * torch.sqrt(
        x * x + y * y + 1e-6
    )

    # Penalize high velocity
    reward -= 0.1 * (
        vx * vx + vy * vy
    )

    # Penalize excessive rotation
    reward -= 0.1 * torch.abs(angle)

    # Small fuel cost
    action_value = action.item()

    if action_value in [1, 3]:
        reward -= 0.03

    elif action_value == 2:
        reward -= 0.3

    return reward.detach().view(1)


def estimate_done(next_state):

    x = next_state[0, 0].item()
    y = next_state[0, 1].item()

    # Very crude initial approximation.
    #
    # This should NOT be considered
    # the actual LunarLander termination rule.

    if abs(x) > 2.0:
        return True

    if y < -1.0:
        return True

    return False

reward_list = []
episode_durations = []

for episode in range(num_episodes):

    print(
        f"Episode {episode} | "
        f"real={len(real_memory)} | "
        f"model={len(model_memory)}"
    )

    state, info = env.reset()

    state = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    total_reward = 0.0

    for t in count():

        action = select_action(state)

        next_state, reward, terminated, truncated, info = (
            env.step(action.item())
        )

        done = (
            terminated
            or truncated
        )

        reward_tensor = torch.tensor(
            [reward],
            dtype=torch.float32,
            device=device
        )

        next_state_tensor = torch.tensor(
            next_state,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        # REAL TRANSITION
        real_memory.push(
            state,
            action,
            next_state_tensor,
            reward_tensor,
            done
        )

        state = next_state_tensor

        total_reward += reward

        # NORMAL DQN UPDATE
        loss = optimize_dqn()

        if loss is not None:
            writer.add_scalar(
                "Loss/DQN",
                loss,
                episode
            )

        # TARGET NETWORK UPDATE
        for target_param, main_param in zip(
            target_net.parameters(),
            policy_net.parameters()
        ):

            target_param.data.copy_(
                TAU * main_param.data
                + (1 - TAU)
                * target_param.data
            )

        if done:
            episode_durations.append(t + 1)
            reward_list.append(total_reward)
            break

    # EPSILON DECAY
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    writer.add_scalar("Reward", total_reward, episode)
    writer.add_scalar("Episode Duration", episode_durations[-1], episode)
    writer.add_scalar("Epsilon", EPSILON, episode)

    # UPDATE DYNAMICS MODEL
    if episode >= 50 and episode % 10 == 0:
        print("Training dynamics model...")

        result = train_dynamics()

        reward_loss = train_reward_model()

        if reward_loss is not None:
            print(
                f"Reward model loss: "
                f"{reward_loss:.6f}"
            )

            writer.add_scalar(
                "RewardModel/Loss",
                reward_loss,
                episode
            )

        if result is not None:
            print("Dynamics:", result)

            print("Dynamics training:")
            print(f" total loss : {result['loss']:.6f}")
            print(f" data loss : {result['data_loss']:.6f}")
            print(f" physics loss : {result['physics_loss']:.6f}")

            # Evaluate one-step prediction accuracy
            dynamics_errors = evaluate_dynamics( dynamics_model, real_memory, device, batch_size=1024, )

            print("Dynamics prediction error:")
            print( f" overall MSE : " f"{dynamics_errors['overall_mse']:.6f}" )
            print( f" overall MAE : " f"{dynamics_errors['overall_mae']:.6f}" )
            print("Per-variable MAE:")

            for name in STATE_NAMES:
                print( f" {name:18s}: " f"{dynamics_errors[name + '_mae']:.6f}" )

            #TensorBoard
            writer.add_scalar( "Dynamics/Error/Overall_MSE", dynamics_errors["overall_mse"], episode, )
            writer.add_scalar( "Dynamics/Error/Overall_MAE", dynamics_errors["overall_mae"], episode, )
            for name in STATE_NAMES:
                writer.add_scalar( f"Dynamics/Error/{name}_MSE", dynamics_errors[f"{name}_mse"], episode, )
                writer.add_scalar( f"Dynamics/Error/{name}_MAE", dynamics_errors[f"{name}_mae"], episode, )

            writer.add_scalar("Dynamics/Loss", result["loss"], episode)
            writer.add_scalar("Dynamics/DataLoss", result["data_loss"], episode)
            writer.add_scalar("Dynamics/PhysicsLoss", result["physics_loss"], episode)

        # Old data generated by model is worse so we clear it
        model_memory.clear()

        # GENERATE SYNTHETIC EXPERIENCE
        generated = generate_model_transitions(num_samples=1000)

        print(f"Generated {generated} "f"synthetic transitions.")

# Close TensorBoard writer
writer.close()

# Save the trained model
torch.save(policy_net.state_dict(), "models/dqn_lunar_lander_200_enhanced.pth")
print("Model saved successfully!")

print("Complete")
env.close()