import subprocess
import sys
import time
import re
import asyncio
import boto3

from daemon3x import Daemon


async def test_async_change_num_nodes():
    # scale
    desired = 2
    try:
        change_num_nodes("mzanur-eks-g4-use1b", "worker-g4-ng", desired)
        result = await asyncio.wait_for(
                asyncio.gather(get_current_nodes(desired)),
                    timeout=60)
    except asyncio.TimeoutError:
        print("command to scale timed out!")
    print(result)



def test_get_current_gns():
    result = get_current_gns('resnet50', 'mzanur-autoscaler', 'r50_elastic_1_delme')
    return result




class ClusterScaler(object):
    def __init__(self,
            model_name,
            cluster_name,
            eks_worker_group,
            bucket_name,
            training_label,
            min_nodes=1,
            max_nodes=8,
            poll_interval=30):
        self._model_name = model_name
        self._cluster_name = cluster_name
        self._eks_worker_group = eks_worker_group
        self._bucket_name = bucket_name
        self._training_label = training_label
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._poll_interval = poll_interval # in seconds


    def get_current_gns(self):
        """
        gns history provides a csv with following information
        current_batch_size,
        current_num_nodes,
        grad_accum_supported,
        scale_one_bs,
        num_grads_accumulated,
        gns
        """
        s3 = boto3.client('s3')
        # bucket_name = 'mzanur-autoscaler'
        # key = 'resnet50/r50_elastic_1_delme/GNS/gns_history.txt'
        key = f'{self._model_name}/{self._training_label}/GNS/gns_history.txt'
        s3_object = s3.get_object(Bucket=self._bucket_name, Key=key)
        body = s3_object['Body']
        text = body.read().decode('utf-8')
        last_line = text.splitlines()[-1]
        print("DUMMY - ", last_line)
        # current_batch_size,current_num_nodes,grad_accum_supported,scale_one_bs,num_grads_accumulated,gns = line.split(',')
        ############


    def change_num_nodes(self, desired_num_nodes):
        """
            eksctl scale nodegroup --cluster=mzanur-eks-g4-use1b --nodes=1 --name=worker-g4-ng
        """
        # output = subprocess.check_output(f"eksctl scale nodegroup --cluster={self._cluster_name} --nodes={desired_num_nodes} --name={self._eks_worker_group}", shell=True)
        print("DUMMY CLUSTER RESIZE COMMAND ISSUED")
        print(output)


    async def get_current_nodes(self, desired_num_nodes):
        while True:
            await asyncio.sleep(10)
            output = subprocess.check_output(f"kubectl get nodes", shell=True)
            # print(output)
            m = re.findall("Ready", str(output))
            ready = 0
            if m:
                print("current ready nodes", len(m))
                ready =  len(m)
                if ready == desired_num_nodes:
                    return ready
            else:
                ready = 0


    def run(self):
        while True:
            with open('/home/ubuntu/workspace/gradstats/eks/yaml/g4/resnet50/elastic/service/debug', 'w') as f:
                print("RUNNNIG", file=f)
            time.sleep(self._poll_interval)
            # check current GNS prediction
            self.get_current_gns()


class Sc4l3rDaemon(Daemon):

    def __init__(self, pid_file):
        super().__init__(pid_file)
        self._scaler = ClusterScaler(
            'resnet50',
            'mzanur-eks-g4-use1b',
            'worker-g4-ng',
            'mzanur-autoscaler',
            'r50_elastic_1_delme',
            min_nodes=1,
            max_nodes=8,
            poll_interval=1)

    def run(self):
        self._scaler.run()

#
#if __name__ == "__main__":
#    # result = asyncio.run(test_async_change_num_nodes())
#    # print(test_get_current_gns())



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




