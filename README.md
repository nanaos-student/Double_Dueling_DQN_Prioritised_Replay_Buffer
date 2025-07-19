# Double_Dueling_DQN_Prioritised_Replay_Buffer
An RL agent to navigate a SUMO road network.

small_scale_data_extraction.py:
Extracts static features from the SUMO network, including internal edges.Launches a continuous SUMO simulation via TraCI.

small_scale_dqn_rl.py:
For every state update, it queries dynamic emissions and vehicle counts
for the current edge and normalizes these by the edge length.
The state vector is composed of:
   [static features] + [normalized distance to destination]
   + [co2_per_meter, co_per_meter, hc_per_meter, nox_per_meter,
      pmx_per_meter, fuel_per_meter, noise_per_meter, vehicle_norm]
A reversal mechanism is included.
Resets the environment for a new episode.
If start_edge and destination_edge are provided, these values are used.
Otherwise, random edges are chosen.
Note: The SUMO simulation remains running.
Executes one transition:
    - Uses action (an index among valid connected edges) to select the next edge.
    - Advances the simulation by one step.
    - Computes the reward and returns (next_state, reward, done, info).
If there are no outgoing edges, attempts a reversal by returning incoming edges from the from-node.
Builds a dueling DQN network with separate streams for value and advantage.
DQN agent using a double dueling network with prioritized replay.
Epsilon is updated per episode according to an exponential decay schedule (approx. 1/e^2).
Runs an inference episode using the trained model in a greedy (epsilon~0) mode.
The function now accepts custom starting and destination edges.
