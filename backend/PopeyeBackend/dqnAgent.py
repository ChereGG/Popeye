import chess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import os
from customEnvironment import CustomEnvironment
from decouple import config

# Configuration paramaters for the whole setup
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
        epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = int(config("max_steps_per_episode"))
mask_moves = True if config("mask_moves") == "True" else False
env = CustomEnvironment(mask_moves=mask_moves)

num_actions = env.action_space.n


def create_q_model():
    initializer = tf.keras.initializers.HeUniform()
    inputs = layers.Input(shape=env.observation_shape)
    flatten = layers.Flatten()(inputs)
    layer1 = layers.Dense(64, activation="relu",kernel_initializer=initializer)(flatten)
    layer2 = layers.Dense(64, activation="relu",kernel_initializer=initializer)(layer1)
    layer3 = layers.Dense(64, activation="relu",kernel_initializer=initializer)(layer2)
    layer4 = layers.Dense(64, activation="relu",kernel_initializer=initializer)(layer3)
    output = layers.Dense(num_actions, activation="linear")(layer4)

    return keras.Model(inputs=inputs, outputs=output)


# The first model makes the predictions for Q-values which are used to
# make a action.
if mask_moves:
    if os.path.isdir("./dqn-agent-masked-moves"):
        model = tf.keras.models.load_model("./dqn-agent-masked-moves")
        model_target = tf.keras.models.load_model("./dqn-agent-masked-moves")
    else:
        model = create_q_model()
        model_target = create_q_model()
else:
    if os.path.isdir("./dqn-agent/"):
        model = tf.keras.models.load_model('./dqn-agent/')
        model_target = tf.keras.models.load_model('./dqn-agent/')
    else:
        model = create_q_model()
        model_target = create_q_model()

optimizer = keras.optimizers.Adam(learning_rate=float(config("learning_rate")))

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# Number of frames for exploration
epsilon_greedy_frames = 10000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 10000
# How often to update the target network
update_target_network = 10
# Using huber loss for stability
loss_function = keras.losses.Huber()
max_reward = None

while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0
    white_to_move = True

    for timestep in range(max_steps_per_episode):
        # env.render()
        if white_to_move:
            # of the agent in a pop up window.
            frame_count += 1

            # Use epsilon-greedy for exploration
            if epsilon > np.random.rand():
                # Take random action
                # action = np.random.choice(num_actions)
                action = np.random.choice(env.get_legal_moves())
            else:
                # Predict action Q-values
                # From environment state
                if mask_moves:
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = model(state_tensor, training=False)[0]
                    ilegal_indices = env.get_ilegal_moves()
                    ilegal_indices = np.expand_dims(ilegal_indices, axis=-1)
                    action_probs = tf.tensor_scatter_nd_update(action_probs,ilegal_indices,[np.NINF]*len(ilegal_indices))
                    action = tf.argmax(action_probs).numpy()
                else:
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = model(state_tensor, training=False)[0]
                    # Take best action
                    action = tf.argmax(action_probs).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target(state_next_sample, training=False)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            # Log details
            template = "episode_reward: {:.2f} at episode {}, frame count {}, epsilon {}"
            print(template.format(episode_reward, episode_count, frame_count, epsilon))
            white_to_move = False

            if done:
                if max_reward is None:
                    max_reward = episode_reward
                    if mask_moves:
                        model.save("./dqn-agent-masked-moves")
                    else:
                        model.save("./dqn-agent")
                elif episode_reward > max_reward:
                    max_reward = episode_reward
                    if mask_moves:
                        model.save("./dqn-agent-masked-moves")
                    else:
                        model.save("./dqn-agent")
                break

        else:
            action = np.random.choice(env.get_legal_moves())
            state_next, reward, done, _ = env.step(action)
            white_to_move=True
            if done:
                break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1
