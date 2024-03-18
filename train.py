import gym
import torch
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("DEVICE : ", device)


from RL_algorithms.Torch.SAC.SAC_ENV import sac 
from RL_algorithms.Torch.SAC.SAC_ENV import core


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',  type=str, default='SpaceRobotState-v0')
    parser.add_argument('--model_path',  type=str, default=None)
    parser.add_argument('--start_steps',  type=int, default=10000)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs = None

    torch.set_num_threads(torch.get_num_threads())
    writer = SummaryWriter(f"logs/train/{args.exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    sac.sac(lambda : gym.make(args.env), model_path=args.model_path ,start_steps=args.start_steps ,actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs,writer=writer)