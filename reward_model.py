import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(8 + 4 + 8, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, state, action, next_state):

        if action.dim() > 1:
            action = action.squeeze(-1)

        action_one_hot = F.one_hot(
            action.long(),
            num_classes=4
        ).float()

        x = torch.cat(
            [
                state,
                action_one_hot,
                next_state
            ],
            dim=1
        )

        return self.net(x)


class RewardTrainer:

    def __init__(
        self,
        model,
        device,
        learning_rate=1e-3,
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.criterion = nn.MSELoss()

    def train_batch(
        self,
        states,
        actions,
        next_states,
        rewards,
    ):

        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)

        predicted_rewards = self.model(
            states,
            actions,
            next_states
        )

        loss = self.criterion(
            predicted_rewards.squeeze(-1),
            rewards
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            10.0
        )

        self.optimizer.step()

        return loss.item()