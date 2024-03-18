import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


import SpaceRobotEnv
# from RL_algorithms.Torch.SAC.SAC_ENV import core 
# from RL_algorithms.Torch.SAC.SAC_Image import core
from RL_algorithms.Torch.SAC.SAC_Image import cnn_model as cnn_core


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def act(model, obs, deterministic=False):
    a, _ = model(obs, deterministic, False)
    return a

def load_model(model_path,env,device):
    # model = core.MLPActorCritic(env.observation_space['observation'], env.action_space, hidden_sizes=[256]*2)
    # model = core.CNNActorCritic(env.action_space, hidden_sizes=[256]*2)
    model_cnn = cnn_core.CNNActor(act_dim=env.action_space.shape[0], activation=nn.ReLU, act_limit=env.action_space.high[0], device=device, hidden_sizes=[256]*2).to(device)

    # Load the state dictionary into the model
    model_cnn.load_state_dict(torch.load(model_path,map_location=device))
    # Ensure the model is in evaluation mode
    model_cnn.eval()
    return model_cnn

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

# Define gym environment
# env_name = "SpaceRobotState-v0"
env_name = "SpaceRobotImage-v0"
env = gym.make(env_name)

# Get the dimensions of the action and observation spaces
# dim_u = env.action_space.shape[0]
# dim_o = env.observation_space["observation"].shape[0]

#reset simulator
observation = env.reset()

# Load the trained model
# model_path = 'model_cnn.pt'
model_path = '/home/anunag/Data/Code/Projects/SLP/space_robot/packages/SpaceRobotEnv/RL_algorithms/Torch/SAC/SAC_Image/model_epoch_9.pt'
model = load_model(model_path,env=env,device=device)

# Inference Loop
for ep in range(5):
    writer = SummaryWriter(f"logs/cnn/dist_error/episode_{ep}")
    observation = env.reset()
    i_step = 0
    while True:
        # env.render(mode='rgb_array') #run without GUI
        env.render() #run with GUI
        # action = model.act(obs=torch.tensor(observation["rawimage"].reshape(1, 3, 64, 64), dtype=torch.float32)) 
        actions = act(model=model,obs=torch.tensor(observation["rawimage"].reshape(1, 3, 64, 64), dtype=torch.float32)).detach().numpy()

        observation, reward, done, _, _ = env.step(actions.reshape(6,))

        distance_error = goal_distance(observation['achieved_goal'], observation['desired_goal'])
        writer.add_scalar("Distance/Error", distance_error, i_step)
        i_step = i_step + 1
        print("Distance/Error :", distance_error)
        if(distance_error < 0.05 or i_step == 100):
            break


# Close the environment
env.close()




# import gym
# import torch

# from RL_algorithms.Torch.SAC.SAC_Image import core
# import SpaceRobotEnv

# test_env = gym.make('SpaceRobotImage-v0')



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"DEVICE : {device} \n")
# ac_kwargs=dict(hidden_sizes=[256]*2)
# # Create actor-critic module and target networks
# actor_critic_agent = core.CNNActorCritic(test_env.action_space, **ac_kwargs).to(device)
# actor_critic_agent.load_state_dict(torch.load("model_epoch_1.pt",map_location=device))
# actor_critic_agent.eval()

# def get_action(observation_, deterministic=False):
#         return actor_critic_agent.act(torch.as_tensor(observation_, dtype=torch.float32), deterministic)

# def test_agent(time_step):
#     print("testing model")
#     for j in range(1):
#         o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
#         o = o["rawimage"].reshape(1, 3, 64, 64)
#         while not(d or (ep_len == 1000)):
#             # Take deterministic actions at test time 
#             action=get_action(o, True).reshape(6,)
        
#             o, r, d, _, _ = test_env.step(action=action)
#             o = o["rawimage"].reshape(1,3, 64, 64)
#             ep_ret += r
#             ep_len += 1
#         # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        
#         print("Test_score", ep_ret, time_step)

# test_agent(1)