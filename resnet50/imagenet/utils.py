import logging
import boto3
from botocore.exceptions import ClientError
import torch
import os

def upload_dir(file_dir, bucket, s3_prefix):
    """Upload a file to an S3 bucket

    :param file_dir: File dir to upload
    :param bucket: Bucket to upload to
    :param s3_prefix: s3 path prefix
    :return: True if file was uploaded, else False
    """

    # Upload the dir
    s3_client = boto3.client('s3')

    for worker_folder in os.listdir(file_dir):
        for file_name in os.listdir(f'{file_dir}/{worker_folder}'):
            try:
                response = s3_client.upload_file(f'{file_dir}/{worker_folder}/{file_name}',
                                                 bucket,
                                                 f'{s3_prefix}/{worker_folder}/{file_name}')
            except ClientError as e:
                logging.error(e)
                return False
    return True

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


def is_global_rank_zero():
    if torch.distributed.get_rank() == 0:
        return True
    return False


def make_path_if_not_exists(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        print(f"{dirpath} created")


def read_s3_textfile(bucket, s3_prefix):
    s3 = boto3.client('s3')
    # bucket_name = 'mzanur-autoscaler'
    # key = 'resnet50/r50_elastic_1_delme/GNS/gns_history.txt'
    # key = f'{model_name}/{training_label}/GNS/gns_history.txt'
    s3_object = s3.get_object(Bucket=bucket, Key=s3_prefix)
    body = s3_object['Body']
    text = body.read().decode('utf-8')
    return text
    
if __name__ == "__main__":
    try:
        upload_file('cluster_detail', 'mzanur-autoscaler', 'resnet50/r50_elastic_1_delme/GNS/cluster_detail')
        txt = read_s3_textfile('mzanur-autoscaler', 'resnet50/r50_elastic_1_delme/GNS/cluster_detail')
        print(txt.splitlines()[-1])
    except ClientError as e:
        print('Something went wrong')
    # upload_dir('/mnt/logs/1622161704', 'mzanur-autoscaler', 'resnet_test')
    pass
