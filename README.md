# gradstats

## To run static cluster setup

See examples [here](./eks/yaml/p3/static/32)

## To run elastic training prototype that uses PT elastic

1. Adjust settings in [gradstats/eks/service/training_scaler_svc.py](./eks/service/training_scaler_svc.py) - this script runs on the same node that launches elastic job (not on compute nodes)
   This script creates a file `node_state`. See an example in `node_state.bk` for the format

2. Example templates used by elastic training can be seen [here](./eks/yaml/g4/resnet50/elastic/)

3. For example changes to training scripts, see [ResNet50 elastic training](./resnet50/imagenet/trainer_ddp_amp_elastic.py)
