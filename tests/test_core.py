import importlib.util
import unittest
import numpy as np
from src.replay_buffer import ReplayBuffer
from src.utils import ROOT, load_config, epsilon_at


class ReplayTests(unittest.TestCase):
    def frame(self, value):
        return np.full((4, 84, 84), value, dtype=np.uint8)

    def test_eviction_and_ownership(self):
        replay = ReplayBuffer(2)
        frame = self.frame(1)
        replay.push(frame, 0, 0, frame, False)
        frame[:] = 99
        self.assertEqual(replay.buffer[0][0][0, 0, 0], 1)
        replay.push(self.frame(2), 1, 1, self.frame(3), False)
        replay.push(self.frame(3), 2, -1, self.frame(4), True)
        states, actions, rewards, next_states, terminals = replay.sample_arrays(2)
        self.assertEqual(len(replay), 2)
        self.assertEqual(set(states[:, 0, 0, 0]), {2, 3})
        self.assertEqual(states.shape, (2, 4, 84, 84))
        self.assertEqual(actions.dtype, np.int64)
        self.assertEqual(rewards.dtype, np.float32)
        self.assertEqual(set(terminals), {0.0, 1.0})

    def test_seeded_sampling(self):
        a, b = ReplayBuffer(10, 7), ReplayBuffer(10, 7)
        for i in range(10):
            for replay in (a, b):
                replay.push(self.frame(i), 0, 0, self.frame(i), False)
        self.assertTrue(np.array_equal(a.sample_arrays(4)[0], b.sample_arrays(4)[0]))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(0)
        replay = ReplayBuffer(1)
        with self.assertRaises(ValueError):
            replay.sample_arrays(1)
        with self.assertRaises(ValueError):
            replay.push(self.frame(1).astype(np.float32), 0, 0, self.frame(1), False)

    def test_configuration(self):
        c = load_config(ROOT / "config/smoke.json")
        self.assertEqual(epsilon_at(0, c), 1.0)
        self.assertAlmostEqual(epsilon_at(c["epsilon_decay_steps"], c), 0.1)
        self.assertAlmostEqual(epsilon_at(100000, c), 0.1)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class LearningTests(unittest.TestCase):
    def test_terminal_target(self):
        import torch
        from src.agent import td_targets
        target = td_targets(torch.tensor([1., 1.]), torch.tensor([[5., 2.], [5., 2.]]),
                            torch.tensor([1., 0.]), 0.99)
        torch.testing.assert_close(target, torch.tensor([1., 5.95]))

    def test_network_update_and_checkpoint(self):
        import tempfile
        import torch
        from src.dqn import DQN
        from src.agent import optimize_model
        torch.set_num_threads(2)
        torch.manual_seed(1)
        model = DQN(4)
        replay = ReplayBuffer(4)
        for i in range(4):
            frame = np.full((4, 84, 84), i, dtype=np.uint8)
            replay.push(frame, i, 1, frame, True)
        before = model.fc2.weight.detach().clone()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00025)
        loss = optimize_model(model, optimizer, replay, 4, 0.99, "cpu")
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.fc2.weight))
        batch = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
        self.assertEqual(model(batch).shape, (2, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/model.pt"
            torch.save(model.state_dict(), path)
            restored = DQN(4)
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(model(batch), restored(batch))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                       ("torch", "gymnasium", "ale_py", "cv2")),
                     "Atari dependencies are not installed")
class EnvironmentTests(unittest.TestCase):
    def test_real_environment_stack(self):
        from src.environment import make_env, as_state
        config = load_config(ROOT / "config/smoke.json")
        env = make_env(config)
        try:
            obs, info = env.reset(seed=7)
            self.assertEqual(as_state(obs).shape, (4, 84, 84))
            self.assertEqual(env.action_space.n, 4)
            obs, reward, terminated, truncated, info = env.step(0)
            self.assertEqual(as_state(obs).dtype, np.uint8)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
