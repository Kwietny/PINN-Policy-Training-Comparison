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