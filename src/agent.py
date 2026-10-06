"""Same-network semi-gradient update from Algorithm 1 of the 2013 paper."""
import torch
import torch.nn.functional as F


def td_targets(rewards, next_q, terminated, gamma):
    return rewards + gamma * next_q.max(dim=1).values * (1 - terminated)


def optimize_model(model, optimizer, replay, batch_size, gamma, device):
    states, actions, rewards, next_states, terminated = replay.sample(batch_size, device)
    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target = td_targets(rewards, model(next_states), terminated, gamma)
    loss = F.mse_loss(current_q, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite TD loss")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
