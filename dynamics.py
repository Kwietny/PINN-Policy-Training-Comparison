import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsResidualDynamics(nn.Module):
    """
    Physics-informed residual dynamics model for LunarLander.

    Input:
        state:  [batch, 8]
        action: [batch]

    Output:
        predicted next state: [batch, 8]
    """

    def __init__(self, dt=1.0):
        super().__init__()

        self.dt = dt

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
            nn.Linear(128, 8),
        )

    def forward(self, state, action):

        # action can arrive as [B, 1]
        if action.dim() > 1:
            action = action.squeeze(-1)

        action_one_hot = F.one_hot(
            action.long(),
            num_classes=4
        ).float()

        x = torch.cat([state, action_one_hot], dim=1)

        # Neural residual
        residual = self.residual_net(x)

        # -------------------------------------------------
        # Basic physics prior
        # -------------------------------------------------

        physics_next = state.clone()

        # x(t+dt) = x(t) + vx(t) dt
        physics_next[:, 0] = (
            state[:, 0]
            + state[:, 2] * self.dt
        )

        # y(t+dt) = y(t) + vy(t) dt
        physics_next[:, 1] = (
            state[:, 1]
            + state[:, 3] * self.dt
        )

        # Everything else is initially assumed unchanged.
        #
        # The neural network learns the correction.

        predicted_next = physics_next + residual

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
            lr=learning_rate
        )

    def train_batch(
        self,
        states,
        actions,
        next_states
    ):

        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)

        predicted_next = self.model(
            states,
            actions
        )

        # -----------------------------
        # Data fitting loss
        # -----------------------------

        data_loss = F.mse_loss(
            predicted_next,
            next_states
        )

        # -----------------------------
        # Physics loss
        # -----------------------------

        p_loss = physics_loss(
            states,
            predicted_next,
            self.model.dt
        )

        # -----------------------------
        # Combined loss
        # -----------------------------

        loss = (
            data_loss
            + self.physics_weight * p_loss
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            10.0
        )

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "data_loss": data_loss.item(),
            "physics_loss": p_loss.item(),
        }

def physics_loss(
    state,
    predicted_next,
    dt=1.0
):
    """
    Penalize violations of basic kinematic relationships.
    """

    expected_x = (
        state[:, 0]
        + state[:, 2] * dt
    )

    expected_y = (
        state[:, 1]
        + state[:, 3] * dt
    )

    loss_x = (
        predicted_next[:, 0]
        - expected_x
    ).pow(2).mean()

    loss_y = (
        predicted_next[:, 1]
        - expected_y
    ).pow(2).mean()

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