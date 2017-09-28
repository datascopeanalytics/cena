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
        print(f'instance already running at {instance_ip}')

    # start_if_not_started(instance)
    # instance_ip = instance.public_ip_address
    return instance_ip


def start_if_not_started(instance):
    if instance.state['Name'] in ['stopped', 'stopping']:
        start_instance(instance)
    else:
        instance_ip = instance.public_ip_address
        print(f'instance already running at {instance_ip}')


def start_instance(instance):
    print(f'Starting instance {instance}...')
    response = instance.start()
    instance.wait_until_running()
    print(f'Instance started at {instance.public_ip_address}')

#
# class EC2Manager(object):
#     def __init__(self, config_path=None, start_if_stopped=True, stop_if_started=False):
#         self.config = config = Config(config_path)
#         self.storm_name, self.storm_enabled = config.storm_name, config.storm_installed
#         self.project_name, self.user_name = config.project_name, config.user_name
#
#         self.ec2_client = boto3.client('ec2')
#         self.ec2_manager = boto3.resource('ec2')
#
#         self.instance_id = self.get_instance_id_from_project_name()
#         self.instance = instance = self.ec2_manager.Instance(self.instance_id)
#         if start_if_stopped:
#             self.start_if_not_started()
#         elif stop_if_started:
#             self.stop_if_not_stopped()
#             return
#
#         self.instance_ip = instance.public_ip_address
#
#         self.public_key_name, self.public_key_path, self.public_key = self.get_public_key()
#
#         if self.storm_name and self.storm_enabled:
#             self.update_storm()
#
#     def start_if_not_started(self):
#         if self.instance.state['Name'] in ['stopped', 'stopping']:
#             self.start_instance()
#         else:
#             self.instance_ip = self.instance.public_ip_address
#             print(f'instance already started at {self.instance_ip}')
#
#     def stop_if_not_stopped(self):
#         state = self.instance.state['Name']
#         if state in ['pending', 'running']:
#             self.stop_instance()
#         else:
#             print('instance already stopped or is stopping!')
#
#     def terminate_instance(self):
#         print(f'alrighty then, terminating instance {self.instance_id}...')
#         self.instance.terminate()
#         self.instance.wait_until_terminated()
#         print('instance terminated')
#
#     def start_instance(self):
#         print(f'Starting instance {self.instance_id}...')
#         response = self.instance.start()
#         self.instance.wait_until_running()
#         print(f'Instance started at {self.instance.public_ip_address}')
#
#     def stop_instance(self):
#         print(f'Stopping instance {self.instance_id}...')
#         response = self.instance.stop()
#         self.instance.wait_until_stopped()
#         print('Instance stopped')
#
#     def update_storm(self):
#         print('Fixin\' up a storm...')
#         storm = Storm()
#         if storm.is_host_in(self.storm_name, regexp_match=True):
#             print('Updating storm profile with latest instance ip')
#             storm_update(
#                 name=self.storm_name,
#                 connection_uri=f'ubuntu@{self.instance_ip}',
#                 id_file=self.public_key_path,
#                 o=['LocalForward=8889 127.0.0.1:8888']
#             )
#         else:
#             print('Creating storm profile')
#             storm_add(
#                         name=self.storm_name,
#                         connection_uri=f'ubuntu@{self.instance_ip}',
#                         id_file=self.public_key_path,
#                         o=['LocalForward=8889 127.0.0.1:8888']
#             )
#         print('Storm updated')
#
#     def get_instance_id_from_project_name(self):
#         filters = [
#             {
#                 'Name': 'tag:created_by',
#                 'Values': [self.user_name]
#             },
#             {
#                 'Name': 'tag:project_name',
#                 'Values': [self.project_name]
#             },
#             {
#                 'Name': 'instance-state-name',
#                 'Values': ['pending', 'running', 'stopping', 'stopped']
#             }
#         ]
#
#         response = self.ec2_client.describe_instances(Filters=filters)
#         instances = response['Reservations']
#         if instances:
#             instance_id = instances[0]['Instances'][0]['InstanceId']
#             return instance_id
#         else:
#             print('No instances found!')
#             return None
#
#     def get_public_key(self):
#         public_key_name = self.instance.key_name
#         public_key_path = os.path.join(os.path.expanduser('~'), '.ssh', f'{public_key_name}.pem')
#         public_key = paramiko.RSAKey.from_private_key_file(public_key_path)
#         print(f'Using public key at {public_key_path}')
#         return public_key_name, public_key_path, public_key
#
#     def resize_instance(self, new_size):
#         self.stop_if_not_stopped()
#         print(f'changing instance to type {new_size}...')
#         self.ec2_client.modify_instance_attribute(InstanceId=self.instance_id, Attribute='instanceType', Value=new_size)
#         print(f'instance type changed')
#         self.start_if_not_started()