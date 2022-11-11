import collections
import copy
import timeit
import pickle
import random
import time
import numpy as np
import gym
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# Using CUDA compatible GPU in training
print("Torch version: ", torch.version.cuda)
print("CUDA available: ", torch.cuda.is_available())
print("CUDA devices: ", torch.cuda.device_count())
print("Current CUDA device: ", torch.cuda.current_device())
# print(torch.cuda.device(0))
print("CUDA device name: ", torch.cuda.get_device_name(0))
print("CUDA device properties: ", torch.cuda.get_device_properties(0))
# print(torch.device('cuda'))
cuda = torch.device('cuda')

torch.backends.cuda.matmul.allow_tf32 = True  # These might give performance gains.
torch.backends.cudnn.allow_tf32 = True

""" Environments for Pong. Frame skipping reduces potential artefacts from Atari animations by
    advancing the game forward multiple frames at a time.
    Pong - skips 2 to 5 frames at a time.
    PongDeterministic - Skips 4 frames at a time. This one is used for this experiment.
    PongNoFrameskip - Doesn't skip any frames. """
env = gym.make('PongDeterministic-v4')

print(env.action_space)
print(env.unwrapped.get_action_meanings())

# Hyperparameters.
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 500_000  # 1 Million step long replay memory didn't fit RAM, so using 500_000 long instead.
TARGET_NET_UPDATE_FREQ = 10_000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
# SQUARED_GRADIENT_MOMENTUM = 0.95
# MIN_SQUARED_GRADIENT = 0.01
INIT_EXPLORATION = 1  # Initial exploration probability.
FINAL_EXPLORATION = 0.1  # Minimum exploration probability at 1 000 000th step.
EXPLORATION_MAX_STEPS = 1_200_000   # Slightly increased due to curated replay memory.
REPLAY_START_SIZE = 50_000

restart = True

# Neural network for
# class QNetwork(nn.Module):
#     def __init__(self):
#         super(QNetwork, self).__init__()
#         nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#         nn.BatchNorm2d(64),
#         nn.ReLU(),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
#         nn.BatchNorm2d(64),
#         nn.ReLU(),
#         nn.Linear(in_features=3136, out_features=512),
#         nn.ReLU(),
#         nn.Linear(in_features=512, out_features=3)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.flatten(x, start_dim=1)
#         x = F.relu(self.fc1(x))
#         x = self.output(x)
#         return x


Qnet = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding='valid', bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding='valid', bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='valid', bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Flatten(start_dim=1),
    nn.Linear(in_features=3136, out_features=512),
    nn.ReLU(),
    nn.Linear(in_features=512, out_features=3),
)


def frame_preprocessing(frame):
    frame = frame[34:194]  # Crop image to 160x160 game area (box).
    # Convert RGB to grayscale (same as luma channel Y' YUV colorspace).
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))  # Rescale to 84x84.
    return frame


def calculate_epsilon(time_steps):
    if time_steps >= EXPLORATION_MAX_STEPS:
        epsilon = FINAL_EXPLORATION
    else:
        epsilon = INIT_EXPLORATION - ((INIT_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION_MAX_STEPS) * time_steps

    return epsilon


def predict_action_rewards(state_t0, epsilon):
    if len(replay_memory) < REPLAY_START_SIZE:
        # Possible actions in Pong: 0 = No action, 2 = Move paddle up, 3 = Move paddle down.
        action = random.choice([0, 2, 3])
    else:
        if epsilon >= np.random.uniform(0, 1):
            action = random.choice([0, 2, 3])
        else:
            # Make 4-frame stack into tensor.
            state_t0_tensor = torch.tensor(data=state_t0, dtype=torch.float32, device=cuda)
            # Change order of layers (84 x 84 x 4) -> (4 x 84 x 84), add 1 layer -> (1 x 4 x 84 x 84).
            state_t0_tensor = state_t0_tensor.permute(2, 0, 1).unsqueeze(0)
            predicted_rewards = Qnet(state_t0_tensor)
            # Tensor index 0 = action 0 (no action), idx 1 = 2 (move up), idx 2 = 3 (move down).
            # Tensor index 0 = action 0 (no action), idx 1 = 2 (move up), idx 2 = 3 (move down).
            action_index = predicted_rewards.argmax().tolist()
            action = [0, 2, 3][action_index]
            match_actions.append(action)  # Statistics variable.

    return action


def get_inactive_frames(env, ball_moving, ball_start_move_frame, ball_move_frames, prev_ball_x_loc, ball_direction, \
                        episode_lost_frame, last_touch_frame):
    """ Cut unnecessary frames by observing the ball position.
        When ball bounces off own paddle, and scores enemy goal, every action after that last bounce
        has zero effect on game outcome. Similarly, after ball passes x-coordinate 190, it goes
        behind own paddle and again every action after this is ineffectual. Also after game reset
        there are some frames after ball spawns before ball or players can move.
        These frames are removed to reduce noise in reward allocation.
    """

    ram = env.unwrapped._get_ram()
    new_ball_x_loc = ram[49]  # X-location increases left and decreases right.

    if not ball_moving:
        if new_ball_x_loc != prev_ball_x_loc and prev_ball_x_loc != 1000:
            ball_moving = True
            ball_start_move_frame = match_framecount - 1
    else:
        if ball_move_frames == 1:
            if new_ball_x_loc < prev_ball_x_loc:
                ball_direction = 1
            else:
                ball_direction = 0
        if ball_move_frames > 1:
            if ball_direction == 0 and new_ball_x_loc > 190 and ball_moving:
                episode_lost_frame = match_framecount
            elif ball_direction == 0 and new_ball_x_loc < prev_ball_x_loc:
                last_touch_frame = match_framecount - 1
                ball_direction = 1
            elif ball_direction == 1 and new_ball_x_loc > prev_ball_x_loc:
                ball_direction = 0

    prev_ball_x_loc = new_ball_x_loc
    if ball_moving:
        ball_move_frames += 1

    return ball_moving, ball_start_move_frame, ball_move_frames, prev_ball_x_loc, ball_direction, \
           episode_lost_frame, last_touch_frame


def calculate_reward(replay_memory, episode_memory, episode_lost_frame, last_touch_frame):
    if reward == -1:
        last_frame = episode_lost_frame
    if reward == 1:
        last_frame = last_touch_frame

    # Remove frames that don't effect outcome.
    episode_memory = episode_memory[ball_start_move_frame:last_frame + 1]

    for i, step in enumerate(episode_memory):
        state_t0, chosen_action = step[0], step[1]
        reward_discounted = reward * (DISCOUNT_FACTOR ** (len(episode_memory) - i - 1))
        replay_memory.append((state_t0, chosen_action, reward_discounted))


def train_network(time_6, time_7, time_8, loss_stats):
    time_6_start = timeit.default_timer()
    # Deque might not be most optimal for this since of need for random sampling?
    # replay_list = list(replay_memory)
    minibatch = random.sample(replay_memory, MINIBATCH_SIZE)

    time_6 += timeit.default_timer() - time_6_start

    time_7_start = timeit.default_timer()
    batch_state_t0 = np.array([x[0] for x in minibatch])
    chosen_actions = [x[1] for x in minibatch]
    target_rewards = [x[2] for x in minibatch]
    batch_state_t0_tensor = torch.tensor(data=batch_state_t0, dtype=torch.float32, device=cuda)

    # Change order of layers (32, 84, 84, 4) -> (32, 4, 84, 84).
    batch_state_t0_tensor = batch_state_t0_tensor.permute(0, 3, 1, 2)
    predicted_rewards = Qtarget(batch_state_t0_tensor).tolist()

    predicted_action_rewards = []
    for i, predicted_values in enumerate(predicted_rewards):
        chosen_action = chosen_actions[i]
        if chosen_action == 0:
            predicted_action_rewards.append(predicted_values[0])
        elif chosen_action == 2:
            predicted_action_rewards.append(predicted_values[1])
        elif chosen_action == 3:
            predicted_action_rewards.append(predicted_values[2])

    time_7 += timeit.default_timer() - time_7_start

    time_8_start = timeit.default_timer()

    predicted_action_rewards = np.array(predicted_action_rewards)
    target_rewards = np.array(target_rewards)
    predicted_rewards_tensor = torch.tensor(data=predicted_action_rewards, dtype=torch.float32, device=cuda,
                                            requires_grad=True)
    target_rewards_tensor = torch.tensor(data=target_rewards, dtype=torch.float32, device=cuda)

    loss = loss_fn(predicted_rewards_tensor, target_rewards_tensor)
    loss_running_avg.append(loss.item())
    loss_stats.append((loss.item(), sum(loss_running_avg) / len(loss_running_avg), time_steps))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(Qnet.parameters(), max_norm=1)  # Gradient culling.
    optimizer.step()
    time_8 += timeit.default_timer() - time_8_start

    if time_steps % 1000 == 0:
        print(f"Loss Running AVG: {sum(loss_running_avg) / len(loss_running_avg)}")
    return time_6, time_7, time_8, loss_stats


# Initialize neural networks and optimizer.
# Qnet = QNetwork()
if not restart:
    with open('training_statistics.pickle', 'rb') as handle:
        match_stats, loss_stats, time_steps = pickle.load(handle)
    Qnet.load_state_dict(torch.load("Qnet_model.pt"))

Qnet.cuda()
Qtarget = copy.deepcopy(Qnet)
Qtarget.cuda()
print("QNET ON CUDA: ", next(Qnet.parameters()).is_cuda)
print("QTARGET ON CUDA: ", next(Qtarget.parameters()).is_cuda)

optimizer = torch.optim.RMSprop(Qnet.parameters(), lr=LEARNING_RATE, momentum=GRADIENT_MOMENTUM)
loss_fn = nn.MSELoss()

# Initialize replay memory and time steps.
time_steps = 0
replay_memory = collections.deque([], REPLAY_MEMORY_SIZE)
prev_frames = collections.deque([], 3)
start_frame = frame_preprocessing(env.reset())
for i in range(0, 3):
    prev_frames.append(start_frame)

# Initialize episode mechanics related variables.
ball_direction = 0
ball_moving = False
ball_move_frames = 0
prev_ball_x_loc = 1000  # Set initial value that's impossible in-game.
ball_start_move_frame = 0
episode_lost_frame = 0
last_touch_frame = 0
episode_memory = []

# Initialize training statistic variables.
time_1, time_2, time_3, time_4, time_5, time_6, time_7, time_8 = 0, 0, 0, 0, 0, 0, 0, 0
opp_score, own_score = 0, 0
match_count = 0
match_framecount = 0
match_actions = []
match_stats = []
loss_stats = []
loss_running_avg = collections.deque(maxlen=1000)
ep_start = timeit.default_timer()

training = True
while training:
    time_1_start = timeit.default_timer()
    time_steps += 1
    #time.sleep(0.2)
    # Render environment every 20 matches to observer training progress.
    if match_count % 20 == 0 and len(replay_memory) >= REPLAY_START_SIZE:
        env.render()
    if time_steps % 1000 == 0:
        print(time_steps, len(replay_memory))

    # Update Qtarget network, and save model and training stats occasionally.
    if time_steps % TARGET_NET_UPDATE_FREQ == 0:
        print("Models saving.")
        torch.save(Qnet.state_dict(), "Qnet_model.pt")
        Qtarget = copy.deepcopy(Qnet)
        Qtarget.cuda()
        with open('training_statistics.pickle', 'wb') as handle:
            pickle.dump([match_stats, loss_stats, time_steps], handle)
        print("Saving done.")

    epsilon = calculate_epsilon(time_steps)
    if time_steps % 1000 == 0:
        print("Epsilon: ", epsilon)
    if match_count % 20 == 0:  # Every 20 matches set epsilon to 0 to observe model without any random actions.
        epsilon = 0

    time_1 += timeit.default_timer() - time_1_start
    time_2_start = timeit.default_timer()

    # Stack 4 last frames to get current state of the game.
    state_t0 = np.dstack((prev_frames[-3], prev_frames[-2], prev_frames[-1]))
    chosen_action = predict_action_rewards(state_t0, epsilon)

    # Ball location tracking for cutting unnecessary frames.
    ball_moving, ball_start_move_frame, ball_move_frames, prev_ball_x_loc, ball_direction, \
    episode_lost_frame, last_touch_frame = get_inactive_frames(env, ball_moving, ball_start_move_frame, ball_move_frames,
                                                prev_ball_x_loc, ball_direction, episode_lost_frame, last_touch_frame)

    frame_t1, reward, done, info = env.step(chosen_action)

    time_3_start = timeit.default_timer()
    frame_t1 = frame_preprocessing(frame_t1)
    time_3 += timeit.default_timer() - time_3_start

    step = (state_t0, chosen_action)
    episode_memory.append(step)
    prev_frames.append(frame_t1)

    time_2 += timeit.default_timer() - time_2_start

    time_4_start = timeit.default_timer()
    if reward == 1 or reward == -1:
        calculate_reward(replay_memory, episode_memory, episode_lost_frame, last_touch_frame)
        episode_memory = []
        ball_move_frames = 0
        prev_ball_x_loc = 1000
        ball_start = False
        game_ongoing = True
    if reward == 1:
        own_score += 1
    if reward == -1:
        opp_score += 1

    if done is True:  # Reset env when match ends at one player reaching 21 points.
        match_count += 1
        print(f"{match_actions.count(0)} : {match_actions.count(2)} : {match_actions.count(3)}")
        match_actions = []
        ep_end = timeit.default_timer()
        match_stats.append((opp_score, own_score, match_framecount, ep_end - ep_start, time_steps))

        print(f"Match score: {opp_score} - {own_score}, runtime: {ep_end - ep_start}, framecount: {match_framecount}")
        own_score, opp_score, match_framecount = 0, 0, 0

        env.close()  # Preventing problems closing window when rendering every 20th match.
        env.reset()
        ep_start = timeit.default_timer()
    time_4 += timeit.default_timer() - time_4_start

    time_5_start = timeit.default_timer()
    if len(replay_memory) >= REPLAY_START_SIZE:
        time_6, time_7, time_8, loss_stats = train_network(time_6, time_7, time_8, loss_stats)
    time_5 += timeit.default_timer() - time_5_start

    if time_steps % 1000 == 0:
        print(f"""Time 1: {time_1:0.0f}. Time 2: {time_2:0.0f}. Time 3: {time_3:0.0f}. Time 4: {time_4:0.0f}.
                 Time 5: {time_5:0.0f}. Time 6: {time_6:0.0f}. Time 7: {time_7:0.0f}. Time 8: {time_8:0.0f}""")

    match_framecount += 1
