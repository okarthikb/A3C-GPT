import lm, os, time, torch, random, wandb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.optim import Adam
from argparse import ArgumentParser


def process(gpu, args, world_size): 
  rank = args.node * args.gpus + gpu
  dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

  torch.manual_seed(69)

  model = lm.LM(args.d, args.nh, args.nl, args.l, args.v).cuda(gpu)
  optimizer = Adam(model.parameters(), lr=args.lr)

  tokens = torch.load('tokens.pt')
  split_size = tokens.shape[0] // world_size
  tokens = tokens[rank * split_size:(rank + 1) * split_size]

  if rank == 0:
    wandb.init(project='A3C LM')
    wandb.run.name = args.name
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n{nparam} parameters\n')

  model.eval()
  for step in range(1, args.steps + 1):
    indices = random.choices(range(0, tokens.shape[0] - args.l - 1), k=args.bs)
    context = torch.stack([tokens[i:i + args.l] for i in indices]).cuda(gpu)

    state = context[:, None, 0].cuda(gpu)

    rewards, values, log_probs = [], [], []

    for i in range(1, args.l):
      probs = model(state)

      if random.random() < args.epsilon:
        action = torch.randint(0, 256, (args.bs,)) 
      else:
        action = Categorical(probs).sample()
      action = action.cuda(gpu)

      action_probs = probs[torch.arange(args.bs), action]
      reward = (action == context[:, i]).float() - action_probs.detach()
      rewards.append(reward)

      value = model(state, False)
      values.append(value)

      log_prob = torch.log(action_probs)
      log_probs.append(log_prob)

      state = torch.cat((state, action[:, None]), 1)

    loss, discounted_return = 0, 0
    for i in range(len(rewards) - 1, -1, -1):
      discounted_return = args.gamma * discounted_return + rewards[i]
      advantage = discounted_return - values[i]
      loss = loss + advantage ** 2 - log_probs[i] * advantage.detach()
    
    loss = loss.sum() 
    loss.backward()
    for p in model.parameters():
      if p.requires_grad:
        dist.all_reduce(p.grad.data)
        p.grad.data /= world_size
    optimizer.step()
    optimizer.zero_grad()

    if rank == 0:
      loss, discounted_return = loss.item(), discounted_return.sum().item()
      wandb.log({'loss': loss, 'reward': discounted_return})
      if step % 10 == 0:
        print(f'step {step}\tloss {loss}\treward {discounted_return}')
  
  if rank == 0:
    torch.save(model.state_dict(), 'model.pt')


# for single node with n GPUs
# python rl.py --nodes 1 --gpus n --name name_of_wandb_run
# have tokens.pt (tensor of token ids)
if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--name', type=str, default='1')
  parser.add_argument('--d', type=int, default=256)
  parser.add_argument('--nh', type=int, default=16)
  parser.add_argument('--nl', type=int, default=16)
  parser.add_argument('--l', type=int, default=64)
  parser.add_argument('--v', type=int, default=1024)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--bs', type=int, default=4)
  parser.add_argument('--gamma', type=float, default=0.999)
  parser.add_argument('--epsilon', type=float, default=0.2)
  parser.add_argument('--steps', type=int, default=1000)
  parser.add_argument('--nodes', type=int, default=1)
  parser.add_argument('--gpus', type=int, default=1)
  parser.add_argument('--node', type=int, default=0)
  args = parser.parse_args()

  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '6969'

  world_size = args.nodes * args.gpus
  mp.spawn(process, args=(args, world_size), nprocs=world_size, join=True)