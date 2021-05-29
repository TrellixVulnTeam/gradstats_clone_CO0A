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


def is_global_rank_zero():
    if torch.distributed.get_rank() == 0:
        return True
    return False

# if __name__ == "__main__":
#     upload_dir('/mnt/logs/1622161704', 'mzanur-autoscaler', 'resnet_test')