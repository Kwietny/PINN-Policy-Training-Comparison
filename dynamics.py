import torch
import torch.nn as nn


class PhysicsResidualDynamics(nn.Module):
    """
    Physics-informed residual dynamics model for LunarLander.

    Input:
        state:  [batch, 8]
        action: [batch]

    Output:
        predicted next state: [batch, 8]
    """

    def __init__(self, dt=1.0, gravity=-0.15):
        super().__init__()

        self.dt = dt
        self.gravity = gravity

        # Action is represented as one-hot:
        # 0 = nothing
        # 1 = left
        # 2 = main
        # 3 = right
        self.residual_net = nn.Sequential(
            nn.Linear(8 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    def forward(self, state, action):

        action_one_hot = torch.nn.functional.one_hot(
            action.long(),
            num_classes=4
        ).float()

        x = torch.cat([state, action_one_hot], dim=1)

        residual = self.residual_net(x)

        # -------------------------------------------------
        # Basic physics prior
        # -------------------------------------------------

        physics_next = state[:, :6].clone()

        x_pos = state[:, 0]
        y_pos = state[:, 1]

        vx = state[:, 2]
        vy = state[:, 3]

        angle = state[:, 4]
        angular_velocity = state[:, 5]

        # Kinematics
        physics_next[:, 0] = x_pos + vx * self.dt
        physics_next[:, 1] = y_pos + vy * self.dt

        # Gravity
        physics_next[:, 2] = vx
        physics_next[:, 3] = vy + self.gravity * self.dt

        # Angular dynamics
        physics_next[:, 4] = (
            angle + angular_velocity * self.dt
        )

        physics_next[:, 5] = angular_velocity

        # Residual correction
        predicted_continuous = physics_next + residual

        # Contact flags are initially learned directly.
        predicted_contact = state[:, 6:8]

        # We could predict them with another network later.
        predicted_next = torch.cat(
            [
                predicted_continuous,
                predicted_contact,
            ],
            dim=1
        )

        return predicted_next

class DynamicsTrainer:

    def __init__(
        self,
        model,
        device,
        learning_rate=1e-3,
        physics_weight=0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.physics_weight = physics_weight

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    def train_step(
        self,
        states,
        actions,
        next_states,
    ):

        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)

        predicted_next = self.model(
            states,
            actions,
        )

        data_loss = (
            predicted_next - next_states
        ).pow(2).mean()

        physics = physics_loss(
            states,
            predicted_next,
            self.model.dt,
        )

        loss = (
            data_loss
            + self.physics_weight * physics
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            10.0,
        )

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "data_loss": data_loss.item(),
            "physics_loss": physics.item(),
        }

def physics_loss(state, predicted_next, dt):
    x = state[:, 0]
    y = state[:, 1]

    vx = state[:, 2]
    vy = state[:, 3]

    next_x = predicted_next[:, 0]
    next_y = predicted_next[:, 1]

    expected_x = x + vx * dt
    expected_y = y + vy * dt

    loss_x = (next_x - expected_x).pow(2).mean()
    loss_y = (next_y - expected_y).pow(2).mean()

    return loss_x + loss_y

def train_dynamics_model(
    dynamics_model,
    replay_buffer,
    device,
    epochs=10,
    batch_size=256,
):

    trainer = DynamicsTrainer(
        dynamics_model,
        device,
        learning_rate=1e-3,
        physics_weight=0.1,
    )

    if len(replay_buffer) < batch_size:
        return

    for epoch in range(epochs):

        states = []
        actions = []
        next_states = []

        batch = replay_buffer.sample(batch_size)

        for transition in batch:
            state, action, next_state, reward, done = transition

            states.append(state)
            actions.append(action)
            next_states.append(next_state)

        states = torch.tensor(
            states,
            dtype=torch.float32,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
        )

        result = trainer.train_step(
            states,
            actions,
            next_states,
        )

    return result