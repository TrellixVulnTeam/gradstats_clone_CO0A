import os
import os.path as osp
import yaml

#TODO: move to dataclass since there is no logic here. Add sane defaults.

class AutoScalerConfig:
    """
    Only configuration that holds for current run. This may be updated in reponse
    to a scaling event.
    """
    def __init__(self, path):
        assert osp.exists(path), f"{path} does not exist."
        config = None
        with open(path, 'r') as f:
            config = yaml.safe_load(f.read())
        autoscaler_config = config['autoscaler']
        adascale_config = config['adascale']
        gns_config = config['gradient_noise_scale']
        self.model_name = autoscaler_config['model_name']
        self.training_label = autoscaler_config['training_label']
        self.log_dir = autoscaler_config['log_dir']
        #self.cluster_state_update_interval = autoscaler_config['cluster_state_update_interval']
        self.s3_bucket = autoscaler_config['s3_bucket']
        self.gradient_accumulation_supported = autoscaler_config['gradient_accumulation_supported']
        self.adjust_gradients_for_accumulation = autoscaler_config['adjust_gradients_for_accumulation']
        self.enable_debug = autoscaler_config['enable_debug']
        self.collect_tensorboard = autoscaler_config['collect_tensorboard']
        self.world_size =  autoscaler_config['world_size']
        self.update_interval = autoscaler_config['update_interval']
        self.precondition_gradients = autoscaler_config['precondition_gradients']
        self.smoothing =  autoscaler_config['smoothing']
        # self.num_gradients_to_accumulate = autoscaler_config['num_gradients_to_accumulate']
        # assert self.num_gradients_to_accumulate >= 1, "Must collect a positive integer"

        # self.enable_adascale = adascale_config['enabled']
        self.aggressive_schedule = adascale_config['aggressive_schedule']
        self.max_grad_norm = adascale_config['max_grad_norm']
        self.is_adaptive = adascale_config['is_adaptive']
        self.use_pt_adam = adascale_config['use_pt_adam']
        # self.enable_gns = gns_config['enabled']
        self.batch_size_upper_limit = gns_config['batch_size_upper_limit']
        self.scale_one_batch_size = gns_config['scale_one_batch_size']
        self.scale_one_world_size = gns_config['scale_one_world_size']


if __name__ == "__main__":
    print(os.getcwd())
    ac = AutoScalerConfig('/Users/mzanur/workspace/gradstats/BERT/autoscaler.yaml')
    print(ac.__dict__)
