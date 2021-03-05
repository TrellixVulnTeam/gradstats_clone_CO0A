import torch
from torch.optim.optimizer import Optimizer
from torch.optim import SGD


class SGDW(Optimizer):
    r"""Implements stochastic gradient descent warm (optionally with momentum) and decoupled weight decay.

    It has been proposed in `Fixing Weight Decay Regularization in Adam <https://arxiv.org/abs/1711.05101>`_.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning <http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf>`_.

    :param params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    :param lr: (float): learning rate
    :param momentum: (float, optional): momentum factor (default: 0)
    :param weight_decay: (float, optional): weight decay (L2 penalty) (default: 0)
    :param dampening: (float, optional): dampening for momentum (default: 0)
    :param nesterov: (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        :param closure: (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                # d_p = p.grad.data
                d_p = p.grad
                orig_p = torch.clone(p).detach()
                if p.grad.is_sparse:
                    msg = (
                        'SGDW does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach() #torch.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # apply update
                p.add_(d_p, alpha=-group['lr'])
                # apply wd on t-1 thetas
                if weight_decay != 0:
                    p.add_(orig_p, alpha=-weight_decay*group['lr'])

        return loss

