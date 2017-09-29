import subprocess
import base64
import boto3
import numpy as np

from cena.settings import API_SERVER_NAME


def encode_image(image):
    image = image.astype(np.uint8)
    return base64.b64encode(image.tobytes()).decode('utf-8')


def decode_image(encoded_str, shape):
    decoded_arr = np.fromstring(base64.b64decode(encoded_str), dtype=np.uint8)
    return decoded_arr.reshape(shape)


def play_mp3(path):
    process = subprocess.Popen(['mpg123', '-q', path])
    process.wait()


def get_api_server_id():
    filters = [
        {
            'Name': 'tag:Name',
            'Values': [API_SERVER_NAME]
        }
    ]

    ec2_client = boto3.client('ec2')

    response = ec2_client.describe_instances(Filters=filters)
    instances = response['Reservations']
    if instances:
        instance_id = instances[0]['Instances'][0]['InstanceId']
        return instance_id
    else:
        raise ValueError('No api server instances found!')


def get_api_server_ip_address():
    api_server_id = get_api_server_id()
    ec2_manager = boto3.resource('ec2')

    instance = ec2_manager.Instance(api_server_id)

    if instance.state['Name'] in ['stopped', 'stopping']:
        start_instance(instance)
        instance_ip = instance.public_ip_address
    else:
        instance_ip = instance.public_ip_address
        print('instance already running at {}'.format(instance_ip))

    # start_if_not_started(instance)
    # instance_ip = instance.public_ip_address
    return instance_ip


def start_if_not_started(instance):
    if instance.state['Name'] in ['stopped', 'stopping']:
        start_instance(instance)
    else:
        instance_ip = instance.public_ip_address
        print('instance already running at {}'.format(instance_ip))


def start_instance(instance):
    print('Starting instance {}...'.format(instance))
    response = instance.start()
    instance.wait_until_running()
    print('Instance started at {}'.format(instance.public_ip_address))
