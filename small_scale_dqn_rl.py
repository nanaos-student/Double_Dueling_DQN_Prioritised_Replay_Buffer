import os
import random
import time
import numpy as np
import joblib
import traci
import sumolib
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Lambda, Subtract, Add
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


# ===============================
# Live SUMO Environment with Dynamic Emissions
# ===============================
class LiveSUMOEnvironment:
    """
    Launches a continuous SUMO simulation via TraCI.
    For every state update, it queries dynamic emissions and vehicle counts
    for the current edge and normalizes these by the edge length.
    The state vector is composed of:
       [static features] + [normalized distance to destination]
       + [co2_per_meter, co_per_meter, hc_per_meter, nox_per_meter,
          pmx_per_meter, fuel_per_meter, noise_per_meter, vehicle_norm]
    A reversal mechanism is included.
    """

    def __init__(self, net_file, sumo_cfg, edge_features_path, max_steps=60):
        self.net_file = net_file
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps

        # Load pre-extracted static edge features and edge IDs
        data = joblib.load(edge_features_path)
        self.edge_features_dict = data['edge_features_dict']
        self.edge_ids = data['edge_ids']

        # Load the network using sumolib for connectivity and geometry
        self.net = sumolib.net.readNet(self.net_file)

        # Determine state dimension:
        # Let F = number of static features; then state = F + 1 (distance) + 8 (dynamic features)
        example_edge = list(self.edge_features_dict.values())[0]
        self.static_feature_size = len(example_edge)
        self.state_size = self.static_feature_size + 9

        # Maximum number of discrete actions
        self.action_size = 8

        # Internal state: current edge, destination edge, and step counter
        self.current_edge = None
        self.destination_edge = None
        self.steps_taken = 0

        # Start SUMO simulation if not already started
        self.sumo_started = False
        self._start_sumo()

        print(f"Live SUMO Environment initialized with {len(self.edge_ids)} edges.")

    def _start_sumo(self):
        if not self.sumo_started:
            sumo_cmd = ["sumo", "-c", self.sumo_cfg, "--no-warnings", "--start"]
            traci.start(sumo_cmd)
            self.sumo_started = True

    def reset(self, start_edge=None, destination_edge=None):
        """
        Resets the environment for a new episode.
        If start_edge and destination_edge are provided, these values are used.
        Otherwise, random edges are chosen.
        Note: The SUMO simulation remains running.
        """
        valid_edges = [e for e in self.edge_ids if not e.startswith(":")]
        if start_edge is None:
            self.current_edge = random.choice(valid_edges)
        else:
            self.current_edge = start_edge
        if destination_edge is None:
            self.destination_edge = random.choice(self.edge_ids)
            while self.destination_edge == self.current_edge:
                self.destination_edge = random.choice(self.edge_ids)
        else:
            self.destination_edge = destination_edge
        self.steps_taken = 0
        return self._get_state()

    def step(self, action):
        """
        Executes one transition:
          - Uses action (an index among valid connected edges) to select the next edge.
          - Advances the simulation by one step.
          - Computes the reward and returns (next_state, reward, done, info).
        """
        connected_edges = self._get_connected_edges(self.current_edge)
        if len(connected_edges) == 0:
            self.steps_taken += 1
            return self._get_state(), -10.0, True, {
                "previous_edge": self.current_edge,
                "current_edge": self.current_edge,
                "steps": self.steps_taken,
                "success": False
            }
        if action >= len(connected_edges):
            next_edge = random.choice(connected_edges)
        else:
            next_edge = connected_edges[action]

        previous_edge = self.current_edge
        self.current_edge = next_edge
        self.steps_taken += 1

        # - Step penalty is -1.0 per step.
        reward = -1.0
        done = False
        success = False
        if self.current_edge == self.destination_edge:
            reward += 100.0  # Bonus for reaching destination.
            done = True
            success = True
        elif self.steps_taken >= self.max_steps:
            reward -= 20.0  # Penalty for exceeding steps.
            done = True

        next_state = self._get_state()
        info = {
            "previous_edge": previous_edge,
            "current_edge": self.current_edge,
            "steps": self.steps_taken,
            "success": success}

        return next_state, reward, done, info

    def _get_connected_edges(self, edge_id):
        """
        Returns a list of outgoing edge IDs for the given edge.
        If there are no outgoing edges, attempts a reversal by returning
        incoming edges from the from-node.
        """
        try:
            edge_obj = self.net.getEdge(edge_id)
            outgoing = edge_obj.getOutgoing()
            connected = [conn.getID() for conn in outgoing]
            if connected:
                return connected
            else:
                from_node = edge_obj.getFromNode()
                incoming = from_node.getIncoming()
                rev_edges = [inc.getID() for inc in incoming if inc.getID() != edge_id]
                if rev_edges:
                    return rev_edges
                else:
                    return [edge_id]
        except KeyError:
            print(f"Edge '{edge_id}' not found in the network.")
            return []

    def _get_state(self):
        """
        Constructs the state vector:
         [static features for current edge] +
         [normalized Euclidean distance to destination] +
         [dynamic emission features:
             co2_per_meter, co_per_meter, hc_per_meter, nox_per_meter,
             pmx_per_meter, fuel_per_meter, noise_per_meter, vehicle_count_norm]
        """
        edge_data = self.edge_features_dict.get(self.current_edge, None)
        if edge_data is None:
            edge_data = defaultdict(float)
        static_vector = [edge_data.get(k, 0.0) for k in edge_data]
        dist = self._compute_distance(self.current_edge, self.destination_edge)
        dist_norm = dist / 1000.0

        try:
            edge_obj = self.net.getEdge(self.current_edge)
            length = edge_obj.getLength()
        except Exception:
            length = 1.0

        # Query SUMO for dynamic features.
        co2 = max(0, traci.edge.getCO2Emission(self.current_edge))
        co = max(0, traci.edge.getCOEmission(self.current_edge))
        hc = max(0, traci.edge.getHCEmission(self.current_edge))
        nox = max(0, traci.edge.getNOxEmission(self.current_edge))
        pmx = max(0, traci.edge.getPMxEmission(self.current_edge))
        fuel = max(0, traci.edge.getFuelConsumption(self.current_edge))
        noise = max(0, traci.edge.getNoiseEmission(self.current_edge))
        veh_count = traci.edge.getLastStepVehicleNumber(self.current_edge)

        co2_per_meter = co2 / length if length > 0 else 0.0
        co_per_meter = co / length if length > 0 else 0.0
        hc_per_meter = hc / length if length > 0 else 0.0
        nox_per_meter = nox / length if length > 0 else 0.0
        pmx_per_meter = pmx / length if length > 0 else 0.0
        fuel_per_meter = fuel / length if length > 0 else 0.0
        noise_per_meter = noise / length if length > 0 else 0.0
        vehicle_norm = veh_count / 10.0

        dynamic_vector = [
            co2_per_meter, co_per_meter, hc_per_meter, nox_per_meter,
            pmx_per_meter, fuel_per_meter, noise_per_meter, vehicle_norm
        ]

        state = np.array(static_vector + [dist_norm] + dynamic_vector, dtype=np.float32)
        return state

    def _compute_distance(self, edge_a, edge_b):
        try:
            edge_obj_a = self.net.getEdge(edge_a)
            edge_obj_b = self.net.getEdge(edge_b)
            shape_a = edge_obj_a.getShape()
            shape_b = edge_obj_b.getShape()
            xa, ya = shape_a[len(shape_a) // 2]
            xb, yb = shape_b[len(shape_b) // 2]
            return np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
        except Exception:
            return 0.0


# ===============================
# Prioritized Replay Buffer
# ===============================
class PrioritizedReplayBuffer:
    """
    Experience replay buffer with prioritized sampling based on TD error.
    """

    def __init__(self, capacity=2000, alpha=0.6, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = []
        self.position = 0

    def add(self, transition):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], []
        scaled = np.array(self.priorities) ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        return samples, indices, probs[indices]

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.epsilon

    def __len__(self):
        return len(self.buffer)


# ===============================
# Double Dueling DQN Network Builder
# ===============================
def build_dueling_dqn(state_size, action_size, learning_rate=1e-4):
    """
    Builds a dueling DQN network with separate streams for value and advantage.
    """
    inputs = Input(shape=(state_size,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    value_fc = Dense(64, activation='relu')(x)
    value = Dense(1, activation='linear')(value_fc)
    adv_fc = Dense(64, activation='relu')(x)
    advantage = Dense(action_size, activation='linear')(adv_fc)
    adv_mean = Lambda(lambda a: K.mean(a, axis=1, keepdims=True))(advantage)
    adv_adjusted = Subtract()([advantage, adv_mean])
    q_values = Add()([value, adv_adjusted])
    model = Model(inputs=inputs, outputs=q_values)
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model


# ===============================
# DQN Agent
# ===============================
class DQNAgent:
    """
    DQN agent using a double dueling network with prioritized replay.
    Epsilon is updated per episode according to an exponential decay schedule (approx. 1/e^2).
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=2000)
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = None  # Update epsilon per episode explicitly.

        self.learning_rate = 1e-4
        self.update_target_frequency = 100
        self.batch_size = 32

        self.model = build_dueling_dqn(state_size, action_size, self.learning_rate)
        self.target_model = build_dueling_dqn(state_size, action_size, self.learning_rate)
        self.update_target_model()

        self.train_steps = 0
        self.q_values_log = []
        self.episode_rewards = []
        self.episode_epsilon = []
        self.episode_steps = []
        self.success_count = 0

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        q_vals = self.model.predict(np.array([state]), verbose=0)[0]
        valid_qs = [(act, q_vals[act]) for act in valid_actions]
        return max(valid_qs, key=lambda x: x[1])[0]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch, indices, _ = self.memory.sample(self.batch_size)
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        target_qs = self.model.predict(states, verbose=0)
        next_qs = self.target_model.predict(next_states, verbose=0)

        td_errors = []
        for i in range(self.batch_size):
            if dones[i]:
                target_val = rewards[i]
            else:
                target_val = rewards[i] + self.gamma * np.max(next_qs[i])
            td_error = target_val - target_qs[i, actions[i]]
            td_errors.append(td_error)
            target_qs[i, actions[i]] = target_val

        self.model.fit(states, target_qs, epochs=1, verbose=0)
        self.memory.update_priorities(indices, td_errors)
        avg_q = np.mean(target_qs)
        self.q_values_log.append(avg_q)
        self.train_steps += 1
        if self.train_steps % self.update_target_frequency == 0:
            self.update_target_model()


# ===============================
# Main Training & Inference Script
# ===============================
def train_dqn_agent_main(net_file, sumo_cfg, edge_features_path, episodes=200):
    env = LiveSUMOEnvironment(net_file, sumo_cfg, edge_features_path, max_steps=60)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    total_episodes = episodes

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps_this_episode = 0
        success_episode = False

        while not done:
            connected_edges = env._get_connected_edges(env.current_edge)
            valid_actions = list(range(len(connected_edges)))
            action = agent.act(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps_this_episode += 1
            print(f"Step {info['steps']}: {info['previous_edge']} -> {info['current_edge']}, Reward: {reward:.2f}")
            if info.get('success', False):
                success_episode = True
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(steps_this_episode)
        # Update epsilon per episode using exponential decay: epsilon = max(epsilon_min, exp(-2*(episode+1)/total_episodes))
        agent.epsilon = max(agent.epsilon_min, np.exp(-2 * (e + 1) / total_episodes))
        agent.episode_epsilon.append(agent.epsilon)
        if success_episode:
            agent.success_count += 1

        print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward:.2f}, Steps: {steps_this_episode}, "
              f"Success: {success_episode}, Epsilon: {agent.epsilon:.3f}")

    print(f"Number of successful episodes: {agent.success_count}")
    plot_results(agent)

    # Save the trained model.
    agent.model.save("trained_dqn_model.keras")
    print("Trained model saved as 'trained_dqn_model.keras'.")

    custom_start_edge = "E0"  # Start
    custom_destination_edge = "E0.25"  # Destination
    path = run_inference(env, agent, custom_start_edge, custom_destination_edge)
    print("Predicted Path from inference:", path)
    return agent


def run_inference(env, agent, start_edge, destination_edge):
    """
    Runs an inference episode using the trained model in a greedy (epsilon~0) mode.
    The function now accepts custom starting and destination edges.
    """
    # Set epsilon to ~0 for greedy policy.
    agent.epsilon = 0.05
    state = env.reset(start_edge=start_edge, destination_edge=destination_edge)
    path = [env.current_edge]
    done = False
    while not done:
        connected_edges = env._get_connected_edges(env.current_edge)
        valid_actions = list(range(len(connected_edges)))
        q_vals = agent.model.predict(np.array([state]), verbose=0)[0]
        valid_qs = [(act, q_vals[act]) for act in valid_actions]
        action = max(valid_qs, key=lambda x: x[1])[0]
        next_state, reward, done, info = env.step(action)
        path.append(info['current_edge'])
        state = next_state
    return path


def plot_results(agent):
    # Compute cumulative rewards
    cum_rewards = np.cumsum(agent.episode_rewards)

    # Figure 1: Avg Q-values & Cumulative Reward
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(agent.q_values_log)
    ax1.set_title("Average Q-Values Over Training")
    ax1.set_xlabel("Training Steps (Batches)")
    ax1.set_ylabel("Average Q-Value")

    ax2.plot(cum_rewards)
    ax2.set_title("Cumulative Reward Over Episodes")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cumulative Reward")

    fig1.tight_layout()

    # Figure 2: Episode Rewards, Epsilon, Steps
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(agent.episode_rewards)
    axes[0].set_title("Episode Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")

    axes[1].plot(agent.episode_epsilon)
    axes[1].set_title("Epsilon Over Episodes")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Epsilon")

    axes[2].plot(agent.episode_steps)
    axes[2].set_title("Steps Per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Steps")

    fig2.tight_layout()
    plt.show()



def main():
    net_file = "small_scale_for_rl.net.xml"
    sumo_cfg = "small_scale_for_rl.sumocfg"
    edge_features_path = "small_scale_extracted_edge_features.pkl"
    agent = train_dqn_agent_main(net_file, sumo_cfg, edge_features_path, episodes=200)


if __name__ == "__main__":
    main()
