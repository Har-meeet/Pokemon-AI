from Emu import Emu
from DQN import DQNAgent
import matplotlib.pyplot as plt
import time

# Initialize Environment and Agent
game_path = "Pokemon-Game\\Pokemon Red.gb"
save_path = "Pokemon-Game\\Starting.state"
env = Emu(game_path, save_path, ticks=[50, 50])

state_dim = 4  # (X, Y, HP, Money)
action_dim = len(env.actions)
agent = DQNAgent(state_dim, action_dim)

num_episodes = 1000
rewards_history = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        env.check_state()
        valid_actions = env.get_valid_action_indices(state)
        action = agent.select_action_from_mask(state, valid_actions)
        print(f"Action: {env.actions_names[action]}")
        
        next_state, reward, done = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay()  # Train on minibatch

    agent.update_target_model()
    rewards_history.append(total_reward)  # Track reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Plot training progress
plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.show()

env.close()