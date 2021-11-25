import logging
import time
from typing import Tuple

import click
import numpy as np
import requests
import torch
from attr import attrs, fields_dict

from FedML.fedml_api.data_preprocessing import LocalDataset, MNISTDataLoader, ShakespeareDataLoader, \
    Cifar100DatasetLoader, Cinic10DatasetLoader, Cifar10DatasetLoader
from FedML.fedml_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer
from FedML.fedml_api.distributed.fedavg.FedAvgClientManager import FedAVGClientManager
from FedML.fedml_api.distributed.fedavg.MyModelTrainer import MyModelTrainer
from FedML.fedml_api.model import RNNOriginalFedAvg
from FedML.fedml_api.model.cv.mobilenet import mobilenet
from FedML.fedml_api.model.cv.resnet import resnet56
from FedML.fedml_api.model.linear.lr import LogisticRegression


@attrs(auto_attribs=True, cmp=False)
class TrainingTaskArgs:
    dataset_name: str
    data_dir: str
    partition_method: str
    partition_alpha: float
    model: str
    client_num_in_total: int
    client_num_per_round: int
    comm_round: int
    epochs: int
    lr: float
    wd: float
    batch_size: int
    frequency_of_the_test: int
    is_mobile: bool

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in fields_dict(cls).keys()})


def register(server_ip, client_uuid) -> Tuple[int, TrainingTaskArgs]:
    # Sending get request and saving the response as response object
    with requests.post(url=f"{server_ip}/api/register", params=dict(device_id=client_uuid)) as r:
        result = r.json()
    client_id = result['client_id']
    # executorId = result['executorId']
    # executorTopic = result['executorTopic']

    return client_id, TrainingTaskArgs.from_dict(result['training_task_args'])


def init_training_device(process_id, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_id == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device

    process_gpu_dict = {client_index: client_index % gpu_num_per_machine for client_index in range(fl_worker_num)}

    logging.info(process_gpu_dict)
    device = torch.device(f"cuda:{process_gpu_dict[process_id - 1]}" if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


def load_data(tt_args: TrainingTaskArgs) -> LocalDataset:
    logging.info(f"load_data. dataset_name = {tt_args.dataset_name}")

    dataloader_cls = dict(
        mnist=MNISTDataLoader,
        shakespeare=ShakespeareDataLoader,
        cifar100=Cifar100DatasetLoader,
        cinic10=Cinic10DatasetLoader,
    ).get(tt_args.dataset_name, Cifar10DatasetLoader)

    dataloader_kwargs = dict(data_dir=tt_args.data_dir, batch_size=tt_args.batch_size)
    if tt_args.dataset_name not in {'mnist', 'shakespeare'}:
        dataloader_kwargs.update(partition_method=tt_args.partition_method, partition_alpha=tt_args.partition_alpha,
                                 client_number=tt_args.client_num_in_total)

    dataloader = dataloader_cls(**dataloader_kwargs)

    dataset = dataloader.load_partition_data()

    if tt_args.client_num_in_total != dataset.client_num:
        logging.warning(f'Updating client_num_in_total from {tt_args.client_num_in_total} to {dataset.client_num}')
        tt_args.client_num_in_total = dataset.client_num

    return dataset


def create_model(args: TrainingTaskArgs, output_dim: int):
    logging.info(f"create_model. model_name = {args.model}, output_dim = {output_dim}")
    if args.model == "lr" and args.dataset_name == "mnist":
        model = LogisticRegression(28 * 28, output_dim)
        args.client_optimizer = "sgd"
    elif args.model == "rnn" and args.dataset_name == "shakespeare":
        model = RNNOriginalFedAvg(28 * 28, output_dim)
        args.client_optimizer = "sgd"
    elif args.model == "resnet56":
        model = resnet56(class_num=output_dim)
    elif args.model == "mobilenet":
        model = mobilenet(class_num=output_dim)
    else:
        raise ValueError(f'Unknown combination for model {args.model} and dataset {args.dataset_name}')
    return model


@click.command()
@click.option('--server_ip', type=str, default='http://127.0.0.1:5000', help='URL address of the FedML server')
@click.option('--client_uuid', type=int, default=0, help='Client identifier number')
@click.option('--gpu_num_per_machine', type=int, default=0, help='Cuda device identifier')
def main(server_ip, client_uuid, gpu_num_per_machine):
    client_id, args = register(server_ip, client_uuid)

    logging.debug(f"args = {args}")
    logging.info(f"client_id = {client_id}")
    logging.info(f"dataset = {args.dataset_name}")
    logging.info(f"model = {args.model}")
    logging.info(f"client_num_per_round = {args.client_num_per_round}")

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(10)

    device = init_training_device(client_id - 1, args.client_num_per_round - 1, gpu_num_per_machine)

    # load data
    dataset = load_data(args)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, output_dim=dataset.output_len)
    model_trainer = MyModelTrainer(model, args.dataset_name, None, args.lr, args.wd, args.epochs)
    model_trainer.set_id(client_id)

    # start training
    trainer = FedAVGTrainer(client_id, dataset, device, args, model_trainer)

    size = args.client_num_per_round + 1
    client_manager = FedAVGClientManager(args, trainer, rank=client_id, size=size, backend="MQTT")
    client_manager.run()
    client_manager.start_training()

    seconds = 5
    logging.info(f'Sleeping for {seconds} seconds')
    time.sleep(seconds)


"""
python mobile_client_simulator.py --client_uuid '0'
python mobile_client_simulator.py --client_uuid '1'
"""
if __name__ == '__main__':
    main()
