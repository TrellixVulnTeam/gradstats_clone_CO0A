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
            cluster_name,
            eks_worker_group,
            model_name,
            bucket_name,
            training_label,
            min_nodes=1,
            max_nodes=8,
            poll_interval=30):
        self._cluster_name = cluster_name
        self._eks_worker_group = eks_worker_group
        self._model_name = model_name
        self._bucket_name = bucket_name
        self._training_label = training_label
        self._min_nodes = min_nodes
        self._max_nodes = max_nodes
        self._poll_interval = poll_interval # in seconds


    def get_current_gns(model_name, bucket_name, training_label):
        """
        gns history provides a csv with following information
        current_batch_size, current_num_nodes, current_grad_accum_factor, gns
        

        """
        s3 = boto3.client('s3')
        # bucket_name = 'mzanur-autoscaler'
        # key = 'resnet50/r50_elastic_1_delme/GNS/gns_history.txt'
        key = f'{model_name}/{training_label}/GNS/gns_history.txt'
        s3_object = s3.get_object(Bucket=bucket_name, Key=key)
        body = s3_object['Body']
        text = body.read().decode('utf-8')
        for line in text.split('\n'):
            if line != "":
                current_state, desired_state = line.split(',')
                current_state = float(current_state.strip()[1:-1]])
                desired_state = float(desired_state[1:-1])
        return int(current_state), int(desired_state)


    def change_num_nodes(self, desired_num_nodes):
        """
            eksctl scale nodegroup --cluster=mzanur-eks-g4-use1b --nodes=1 --name=worker-g4-ng
        """
        output = subprocess.check_output(f"eksctl scale nodegroup --cluster={self._cluster_name} --nodes={desired_num_nodes} --name={self._eks_worker_group}", shell=True)
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
            time.sleep(self._poll_interval)
            # check current GNS prediction
            current, desired = get_current_gns(self._model_name, 'mzanur-autoscaler', 'r50_elastic_1_delme')
             



class Sc4l3rDaemon(Daemon):
    def run(self):
        # Or simply merge your code with MyDaemon.
        your_code = YourCode()
        your_code.run()

#
#if __name__ == "__main__":
#    # result = asyncio.run(test_async_change_num_nodes())
#    # print(test_get_current_gns())



if __name__ == "__main__":
    daemon = Sc4l3rDaemon('/tmp/sc4l3r-daemon.pid')
    if len(sys.argv) == 2:
        if 'start' == sys.argv[1]:
            daemon.start()
        elif 'stop' == sys.argv[1]:
            daemon.stop()
        elif 'restart' == sys.argv[1]:
            daemon.restart()
        else:
            print "Unknown command"
            sys.exit(2)
        sys.exit(0)
    else:
        print "usage: %s start|stop|restart" % sys.argv[0]
        sys.exit(2)




