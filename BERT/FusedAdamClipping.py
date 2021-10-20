from apex.optimizers import FusedLAMB, FusedAdam
import torch
from apex.multi_tensor_apply import multi_tensor_applier
import torch.distributed as dist

class FusedAdamClipping(FusedAdam):

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        
        max_grad_norm = 5.0
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError('FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.')
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedAdam only support fp16 and fp32.')

            # print("### g_16 before clip:", g_16)
            # print("### g_32 before clip:", g_32)
            torch.set_printoptions(profile="full")
            if dist.get_rank() == 0:
                print("### g_16.shape: ", len(g_16))
                for ele in g_16:
                    print("g16_ele ", ele)

                print("### g_32.shape: ", len(g_32))
                for ele in g_32:
                    print("g32_ele ", ele)
                torch.nn.utils.clip_grad_norm_(g_16, max_grad_norm)
                torch.nn.utils.clip_grad_norm_(g_32, max_grad_norm)
                print("After appliciaton")
                for ele in g_16:
                    print("g16_ele ", ele)
                for ele in g_32:
                    print("g32_ele ", ele)

            # print("### g_16 After clip:", g_16)
            # print("### g_32 After clip:", g_32)

            if(len(g_16) > 0):
                multi_tensor_applier(self.multi_tensor_adam,
                                     self._dummy_overflow_buf,
                                     [g_16, p_16, m_16, v_16],
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     group['step'],
                                     self.adam_w_mode,
                                     bias_correction,
                                     group['weight_decay'])
            if(len(g_32) > 0):
                multi_tensor_applier(self.multi_tensor_adam,
                                     self._dummy_overflow_buf,
                                     [g_32, p_32, m_32, v_32],
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     group['step'],
                                     self.adam_w_mode,
                                     bias_correction,
                                     group['weight_decay'])


        return loss
