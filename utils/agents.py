import torch as th
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise, GaussianNoise


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, actor_lr=0.01, critic_lr=0.01,
                 discrete_action=True, **kwargs):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.use_cuda = th.cuda.is_available() and kwargs['use_cuda']
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=kwargs['actor_layer'],
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=kwargs['critic_layer'],
                                 constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=kwargs['actor_layer'],
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=kwargs['critic_layer'],
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        if not discrete_action:
            if kwargs['noise'] == 'ou':
                self.exploration = OUNoise(num_out_pol)
            elif kwargs['noise'] == 'gaussian':
                # raise NotImplementedError()
                self.exploration = GaussianNoise(num_out_pol)
            else:
                raise NotImplementedError()
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += th.from_numpy(self.exploration.noise()).type(FloatTensor)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])