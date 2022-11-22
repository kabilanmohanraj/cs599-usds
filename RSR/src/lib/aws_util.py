import logging
import boto3
from botocore.exceptions import ClientError
import io


class S3Manager:
    def __init__(self, key_id, access_key, region):
        self.key_id = key_id
        self.access_key = access_key
        self.region = region

    def create_bucket(self, bucket_name):
        """Create an S3 bucket in a specified region

        If a region is not specified, the bucket is created in the S3 default
        region (us-east-1).

        :param bucket_name: Bucket to create
        :param region: String region to create bucket in, e.g., 'us-west-2'
        :return: True if bucket created, else False
        """

        # Create bucket
        session = boto3.Session(
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.access_key,
            region_name = self.region
        )
        try:
            s3_client = boto3.client('s3')
            if self.region != "us-east-1":
                location = {'LocationConstraint': self.region}
                s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
            else:
                s3_client.create_bucket(Bucket=bucket_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

    def save_data(self, bucket, file_name, data):
        session = boto3.Session(
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.access_key,
            region_name = self.region
        )
        s3 = session.resource('s3')
        s3_object = s3.Object(bucket, file_name)
        s3_object.put(Body=data)

    def get_data(self, bucket, file_name):
        session = boto3.Session(
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.access_key,
            region_name = self.region
        )
        s3 = session.resource('s3')
        obj = s3.Object(bucket, file_name)
        return obj.get()['Body'].read()

    def get_files(self, bucket, filter=None):
        session = boto3.Session(
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.access_key,
            region_name = self.region
        )
        s3 = session.resource('s3')
        my_bucket = s3.Bucket(bucket)
        keys = []
        if filter is None:
            for my_bucket_object in my_bucket.objects.all():
                keys.append(my_bucket_object.key)
        else:
            for my_bucket_object in my_bucket.objects.filter(Prefix=filter):
                keys.append(my_bucket_object.key)
        return keys


class SQSQueueManager:
    def __init__(self, queue_url, region, key_id, access_key):
        self.queue = queue_url
        self.region = region
        self.key_id = key_id
        self.access_key = access_key
    
    def send_message(self,msg_id, msg, zone):
        """
        Sends a message to the specified queue.
        """
        session = boto3.Session(
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.access_key,
            region_name = self.region
        )
        sqs_client = session.resource("sqs")
        queue = sqs_client.get_queue_by_name(QueueName=self.queue)
        try:
            response = queue.send_message(MessageBody=msg, MessageAttributes={
                                                'MsgId': {
                                                    'DataType': 'String',
                                                    'StringValue': msg_id
                                                },
                                                'Zone': {
                                                    'DataType': 'String',
                                                    'StringValue': zone
                                                }
                                            })
        except ClientError:
            print("Unable to send message...")
            raise
        else:
            return response
    
    def recieve_messages(self):
        """
        Sends a message to the specified queue.
        """
        session = boto3.Session(
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.access_key,
            region_name = self.region
        )
        sqs_client = session.resource("sqs")
        queue = sqs_client.get_queue_by_name(QueueName=self.queue)
        try:
            response = queue.receive_messages(AttributeNames=['SentTimestamp'], MessageAttributeNames=['MsgId','Zone'], MaxNumberOfMessages=10, WaitTimeSeconds=10)
        except ClientError:
            print("Unable to recieve message...")
            raise
        else:
            return response