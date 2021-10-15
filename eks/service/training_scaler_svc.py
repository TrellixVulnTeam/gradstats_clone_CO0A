import subprocess
import os
import sys
import time
import re
import asyncio
import boto3
from botocore.exceptions import ClientError

from daemon3x import Daemon


def upload_file(filepath, bucket, s3_prefix):
    """Upload a file to an S3 bucket

    :param filepath: File to upload
    :param bucket: Bucket to upload to
    :param s3_prefix: s3 path prefix
    :return: True if file was uploaded, else False
    """
    # Upload the dir
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(filepath,
                                         bucket,
                                         s3_prefix)
    except ClientError as e:
        logging.error(e)
        return False
    return True

#TODO: separate scaling logic from cluster state (create a state class)

class ClusterScaler(object):
    def __init__(self,
            model_name,
            cluster_name,
            eks_worker_group,
            bucket_name,
            training_label,
            base_yaml=None,
            out_yaml=None,
            nodestate_file=None,
            etcd_addr=None,
            min_nodes=1,
            max_nodes=8,
            gpus_per_node=4,
            poll_interval=30):
        self._model_name = model_name
        self._cluster_name = cluster_name
        self._eks_worker_group = eks_worker_group
        self._bucket_name = bucket_name
        self._training_label = training_label
        self._base_yaml = base_yaml
        self._out_yaml = out_yaml
        self._nodestate_file = nodestate_file
        self._min_nodes = min_nodes
        self._last_scaling_recommendation = min_nodes
        self._last_grad_accum_steps = 1
        self._max_nodes = max_nodes
        self._gpus_per_node = gpus_per_node
        self._etcd_addr = etcd_addr
        self._poll_interval = poll_interval # in seconds
        self._current_num_nodes = self._get_current_ready_nodes()
        if self._current_num_nodes < self._min_nodes:
            # issue a eksctl scaling request
            asyncio.run(self._scale_cluster(self._min_nodes))
        print(f'{self._current_num_nodes} worker nodes available')
        # if node state file exists then training job is running and nothing
        # further needs to be done else we launch a training job
        if nodestate_file and not os.path.exists(nodestate_file):
            # launch training job
            self._launch_training_job(self._min_nodes, rescale_existing=False)
        self._last_scale_timestamp = -1

    async def _scale_cluster(self, desired, timeout=120):
        """
        Issue a eksctl scaling request and wait upto timeout seconds for nodes to be in Ready state
        """
        try:
            self._change_num_nodes(desired)
            result = await asyncio.wait_for(
                                asyncio.gather(self._is_desired_num_nodes_available(desired)),
                                timeout=timeout) # seconds
        except asyncio.TimeoutError:
            print('Command to scale timed out!')
            result = False
        return result


    def _get_current_cluster_state(self):
        """
        gns history provides a csv with following information
        current_bs,
        current_num_workers (1/GPU),
        grad_accum_supported,
        scale_one_bs,
        num_grads_accumulated,
        gns
        timestamp
        """
        try:
            s3 = boto3.client('s3')
            # bucket_name = 'mzanur-autoscaler'
            # key = 'resnet50/r50_elastic_1_delme/GNS/gns_history.txt'
            key = f'{self._model_name}/{self._training_label}/GNS/gns_history.txt'
            s3_object = s3.get_object(Bucket=self._bucket_name, Key=key)
            body = s3_object['Body']
            text = body.read().decode('utf-8')
            last_line = text.splitlines()[-1]
            # (3840, 60, True, 256, 1, 5865, 1633599616)
            current_bs,current_num_workers,grad_accum_supported,scale_one_bs,num_grads_accumulated,gns,timestamp = last_line.split(',')
            current_bs = int(current_bs)
            current_num_workers = int(current_num_workers)
            grad_accum_supported = bool(grad_accum_supported)
            scale_one_bs = int(scale_one_bs)
            num_grads_accumulated = int(num_grads_accumulated)
            gns = int(gns)
            timestamp = int(timestamp)
            return current_bs,current_num_workers,grad_accum_supported,scale_one_bs,num_grads_accumulated,gns,timestamp
        except Exception as e:
            # print("Something went wrong while fetching GNS information from autoscaler", e)
            return None


    def _launch_training_job(self, num_nodes, rescale_existing=False):
        """
        This is used if we want to 1. start a training job 2. scale a training job
        Scenario 1 will use kubectl create --save-config
        Scenario 2 will apply cluster changes on top of scenario 1 config
        """
        result = False
        if not rescale_existing:
            # scenario 1
            # a. start num_nodes in cluster (c-tor)
            # b. prepare training yaml
            self._prepare_training_job_yaml(self._min_nodes)
            # c. launch kubectl job
            output = subprocess.check_output(f"kubectl create --save-config -f {self._out_yaml}", shell=True)
            print("Launched training job...")
            time.sleep(60)
            result = True
        else:
            # scenario 2 - check nodes in cluster, if scaling down, prepare yaml, apply yaml
            # TODO: check which nodes are not running job or etcd and bring them down gracefully
            # if scaling up then, prepare yaml, provision new nodes, on successful provision apply yaml
            result = asyncio.run(self._scale_cluster(num_nodes, timeout=3600)) # provisioning (cold-start) can take significantly long - JACUZZI!
            if result:
                self._prepare_training_job_yaml(num_nodes)
                output = subprocess.check_output(f"kubectl apply -f {self._out_yaml}", shell=True)
                print("Applied new configuration to scale training job...")
            else:
                print("Scaling cluster failed... keeping cluster as before")
        # output = subprocess.check_output(f"kubectl get pods -n elastic-job", shell=True)
        # print(output)
        return result


    def _change_num_nodes(self, desired_num_nodes):
        """
        eksctl scale nodegroup --cluster=mzanur-eks-g4-use1b --nodes=1 --name=worker-g4-ng
        """
        output = subprocess.check_output(f"eksctl scale nodegroup --cluster={self._cluster_name} --nodes={desired_num_nodes} --name={self._eks_worker_group}", shell=True)
        print("CLUSTER RESIZE COMMAND ISSUED")
        print(output)


    def _get_current_ready_nodes(self):
        output = subprocess.check_output(f"kubectl get nodes", shell=True)
        m = re.findall("Ready", str(output))
        if m:
            return len(m)
        else:
            return 0


    async def _is_desired_num_nodes_available(self, desired_num_nodes):
        """
        This function should be called from an external timed executor else
        this will loop endlessly!
        """
        while True:
            await asyncio.sleep(1)
            ready = self._get_current_ready_nodes()
            if ready == desired_num_nodes:
                return True


    def _prepare_training_job_yaml(self, desired_nodes):
        """
        sed correct num_replicas in yaml template
        """
        with open(self._out_yaml, 'w') as f:
            subprocess.call(['sed', f's/{{{{num_replicas}}}}/{desired_nodes}/; s/{{{{etcd_server}}}}/{self._etcd_addr}/', self._base_yaml], stdout=f)


    def _get_scaling_recommendation(self, current_cluster_state):
        trigger_scaling = False
        new_grad_accum_steps = 1
        current_bs,current_num_workers,grad_accum_supported,scale_one_bs,num_grads_accumulated,gns,timestamp = current_cluster_state
        if self._last_scale_timestamp == timestamp:
            return False, 0, 0
        self._last_scale_timestamp = timestamp
        desired_scaling_factor = min(gns // scale_one_bs, 2 * self._max_nodes) # limit scaling to 2x the nodes
        current_scaling_factor = current_bs // scale_one_bs
        print(f'current_bs={current_bs}, gns={gns}, desired_scaling_factor={desired_scaling_factor}, current_scaling_factor={current_scaling_factor}')
        nodes_required = min(2 * current_scaling_factor, desired_scaling_factor) # increase scale by 2x each time
        if desired_scaling_factor <= current_scaling_factor:
            trigger_scaling = False #TODO: downscaling cluster not supported initially
        elif nodes_required > self._max_nodes:
            nodes_required = self._max_nodes
            if current_scaling_factor < self._max_nodes:
                # first fill up nodes that are available
                trigger_scaling = True
                new_grad_accum_steps = 1
            elif grad_accum_supported:
                if desired_scaling_factor % self._max_nodes != 0:
                    # we only support accumulation with multiples of per gpu batch size
                    trigger_scaling = False
                    new_grad_accum_steps = 1
                else:
                    trigger_scaling = True
                    new_grad_accum_steps = int(desired_scaling_factor/self._max_nodes)
                    if self._last_grad_accum_steps >= new_grad_accum_steps:
                        trigger_scaling = False
                    else:
                        self._last_grad_accum_steps = new_grad_accum_steps
            else:
                trigger_scaling = False
        else:
            # until we exhaust nodes do not add gradient accumulation
            trigger_scaling = True
        print("Scaling recommendation:", trigger_scaling, nodes_required, new_grad_accum_steps)
        # sometimes when we recommended scaling the elastic setup does not
        # respond as fast as desired and we end up issuing multiple requests which
        # leads us to inconsistent training state
        if trigger_scaling:
            # first check if we achieved the last rescale target
            if current_scaling_factor != self._last_scaling_recommendation:
                # disable additional scaling until we hit the previous target
                trigger_scaling = False
                print("Canceling further scaling since previous scale request has not been completed")
            else:
                self._last_scaling_recommendation = nodes_required
        return trigger_scaling, nodes_required, new_grad_accum_steps



    ####### MAIN SERVICE LOOP #######
    def run(self):
        skip_count = 0
        while True:
            time.sleep(self._poll_interval)
            # check current GNS prediction
            current_cluster_state = self._get_current_cluster_state()
            if current_cluster_state:
                print("Current cluster state:", current_cluster_state)
                trigger_scaling, nodes_required, new_grad_accum_steps = self._get_scaling_recommendation(current_cluster_state)
                if trigger_scaling:
                    with open(self._nodestate_file, 'w') as f:
                        print(f'{nodes_required},{new_grad_accum_steps}', file=f)
                    # push to S3
                    prefix = f'{self._model_name}/{self._training_label}/GNS/node_state'
                    upload_file(self._nodestate_file, 'mzanur-autoscaler', prefix)
                    if self._get_current_ready_nodes() < nodes_required:
                        result = self._launch_training_job(nodes_required, rescale_existing=True)
                    else:
                        print("Nodes already available skipping EKS provisioning")
                        self._prepare_training_job_yaml(nodes_required)
                        output = subprocess.check_output(f"kubectl apply -f {self._out_yaml}", shell=True)
                        print("Applied new configuration to scale training job...")
                else:
                    print("No rescale triggered:", trigger_scaling, nodes_required, new_grad_accum_steps)



class Sc4l3rDaemon(Daemon):

    def __init__(self, pid_file):
        super().__init__(pid_file, debug=True)
        self._scaler = ClusterScaler(
            'resnet50',
            'mzanur-eks-g4-use1b',
            'worker-g4-ng',
            'mzanur-autoscaler',
            'r50_elastic_fix_1',
            base_yaml='/home/ubuntu/workspace/gradstats/eks/yaml/g4/resnet50/elastic/r50_elastic_training_job_template.yaml',
            out_yaml='/home/ubuntu/workspace/gradstats/eks/yaml/g4/resnet50/elastic/r50_elastic_training_job.yaml',
            nodestate_file='/home/ubuntu/workspace/gradstats/eks/service/node_state',
            etcd_addr="10.100.9.93",
            min_nodes=2, # 1, #FIXME: S=1 gns is broken(?)
            max_nodes=16,
            gpus_per_node=4,
            poll_interval=900) # check for cluster state every 15 mins (this also controls cluster resize frequency)


    def run(self):
        self._scaler.run()


if __name__ == "__main__":
    daemon = Sc4l3rDaemon('/tmp/scaler-daemon.pid')
    if len(sys.argv) == 2:
        if 'start' == sys.argv[1]:
            daemon.start()
        elif 'stop' == sys.argv[1]:
            daemon.stop()
        elif 'restart' == sys.argv[1]:
            daemon.restart()
        else:
            print("Unknown command")
            sys.exit(2)
        sys.exit(0)
    else:
        print("usage: %s start|stop|restart" % sys.argv[0])
        sys.exit(2)

