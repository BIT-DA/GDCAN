import os
import os.path as osp
from loss import *
from torch.utils.data import DataLoader
import random
import argparse
import os
import time
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pre_process as prep
from pre_process import transforms
from data_list import ImageList
import lr_schedule
from logger import Logger
import numpy as np
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import warnings
import torchvision
from network import gdcca_resnet50, gdcca_resnet101

cudnn.benchmark = True
cudnn.deterministic = True
warnings.filterwarnings('ignore')


def image_classification_test(loader, base_net, classifier_layer, residual_layer1,
                              residual_layer2, test_10crop, config, num_iter):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    features_base = base_net(inputs[j])
                    features_residual1 = residual_layer1(features_base)
                    residual_total1 = features_base + features_residual1
                    outputs_old = classifier_layer(residual_total1)
                    outputs_residual = residual_layer2(outputs_old)
                    outputs_new = outputs_residual + outputs_old

                    outputs.append(nn.Softmax(dim=1)(outputs_new))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                features_base = base_net(inputs)
                features_residual1 = residual_layer1(features_base)

                residual_total1 = features_base + features_residual1
                outputs_old = classifier_layer(residual_total1)
                outputs_residual = residual_layer2(outputs_old)
                outputs = outputs_residual + outputs_old

                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float().cpu()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if config['is_writer']:
        config['writer'].add_scalars('test', {'test error': 1.0 - accuracy,
                                              'acc': accuracy * 100.0},
                                     num_iter)

    return accuracy * 100.0


def train(config):
    # pre-process training and test data
    prep_dict = {'source': prep.image_train(**config['prep']['params']),
                 'target': prep.image_train(**config['prep']['params'])}
    if config['prep']['test_10crop']:
        prep_dict['test'] = prep.image_test_10crop(**config['prep']['params'])
    else:
        prep_dict['test'] = prep.image_test(**config['prep']['params'])

    data_set = {}
    dset_loaders = {}
    data_config = config['data']
    data_set['source'] = ImageList(open(data_config['source']['list_path']).readlines(),
                                   transform=prep_dict['source'])
    dset_loaders['source'] = torch.utils.data.DataLoader(data_set['source'],
                                                         batch_size=data_config['source']['batch_size'],
                                                         shuffle=True, num_workers=4, drop_last=True)
    data_set['target'] = ImageList(open(data_config['target']['list_path']).readlines(),
                                   transform=prep_dict['target'])
    dset_loaders['target'] = torch.utils.data.DataLoader(data_set['target'],
                                                         batch_size=data_config['target']['batch_size'],
                                                         shuffle=True, num_workers=4, drop_last=True)
    if config['prep']['test_10crop']:
        data_set['test'] = [ImageList(open(data_config['test']['list_path']).readlines(),
                                      transform=prep_dict['test'][i]) for i in range(10)]
        dset_loaders['test'] = [torch.utils.data.DataLoader(dset, batch_size=data_config['test']['batch_size'],
                                                            shuffle=False, num_workers=4) for dset in data_set['test']]
    else:
        data_set['test'] = ImageList(open(data_config['test']['list_path']).readlines(), transform=prep_dict['test'])
        dset_loaders['test'] = torch.utils.data.DataLoader(data_set['test'],
                                                           batch_size=data_config['test']['batch_size'],
                                                           shuffle=False, num_workers=4)

    # set base network, classifier network, residual net
    class_num = config['network']['params']['class_num']
    net_config = config['network']
    if net_config['name'] == '50':
        base_network = gdcca_resnet50()
    elif net_config['name'] == '101':
        base_network = gdcca_resnet101()
    else:
        raise ValueError('base network %s not found!' % (net_config['name']))

    classifier_layer = nn.Linear(2048, class_num)
    # feature residual layer: two fully connected layers
    residual_fc1 = nn.Linear(2048, 2048)
    residual_bn1 = nn.BatchNorm1d(2048)
    residual_fc2 = nn.Linear(2048, 128)
    residual_bn2 = nn.BatchNorm1d(128)
    residual_fc3 = nn.Linear(128, 2048)
    classifier_layer.weight.data.normal_(0, 0.01)
    classifier_layer.bias.data.fill_(0.0)
    residual_fc1.weight.data.normal_(0, 0.005)
    residual_fc1.bias.data.fill_(0.1)
    residual_fc2.weight.data.normal_(0, 0.005)
    residual_fc2.bias.data.fill_(0.1)
    residual_fc3.weight.data.normal_(0, 0.005)
    residual_fc3.bias.data.fill_(0.1)
    feature_residual_layer = nn.Sequential(residual_fc2, nn.ReLU(), residual_fc3)

    # class residual layer: two fully connected layers
    residual_fc22 = nn.Linear(classifier_layer.out_features, classifier_layer.out_features)
    residual_bn22 = nn.BatchNorm1d(classifier_layer.out_features)
    residual_fc23 = nn.Linear(classifier_layer.out_features, classifier_layer.out_features)
    residual_fc22.weight.data.normal_(0, 0.005)
    residual_fc22.bias.data.fill_(0.1)
    residual_fc23.weight.data.normal_(0, 0.005)
    residual_fc23.bias.data.fill_(0.1)
    class_residual_layer = nn.Sequential(residual_fc22, nn.ReLU(), residual_fc23)

    base_network = base_network.cuda()
    feature_residual_layer = feature_residual_layer.cuda()
    classifier_layer = classifier_layer.cuda()
    class_residual_layer = class_residual_layer.cuda()
    softmax_layer = nn.Softmax().cuda()

    # set optimizer
    parameter_list = [
        {'params': base_network.parameters(), 'lr_mult': 1, 'decay_mult': 2},
        {'params': classifier_layer.parameters(), 'lr_mult': 10, 'decay_mult': 2},
        {'params': feature_residual_layer.parameters(), 'lr_mult': 0.01, 'decay_mult': 2},
        {'params': class_residual_layer.parameters(), 'lr_mult': 0.01, 'decay_mult': 2}
    ]
    optimizer_config = config['optimizer']
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    schedule_param = optimizer_config['lr_param']
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config['lr_type']]

    # set loss
    class_criterion = nn.CrossEntropyLoss().cuda()
    loss_config = config['loss']
    if 'params' not in loss_config:
        loss_config['params'] = {}

    # train
    len_train_source = len(dset_loaders['source'])
    len_train_target = len(dset_loaders['target'])
    best_acc = 0.0
    since = time.time()
    for num_iter in range(config['max_iter']):
        if num_iter % config['val_iter'] == 0 and num_iter != 0:
            base_network.train(False)
            classifier_layer.train(False)
            feature_residual_layer.train(False)
            class_residual_layer.train(False)
            base_network = nn.Sequential(base_network)
            classifier_layer = nn.Sequential(classifier_layer)
            feature_residual_layer = nn.Sequential(feature_residual_layer)
            class_residual_layer = nn.Sequential(class_residual_layer)
            temp_acc = image_classification_test(loader=dset_loaders, base_net=base_network,
                                                 classifier_layer=classifier_layer,
                                                 residual_layer1=feature_residual_layer,
                                                 residual_layer2=class_residual_layer,
                                                 test_10crop=config['prep']['test_10crop'],
                                                 config=config, num_iter=num_iter
                                                 )
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = {'base': base_network.state_dict(), 'classifier': classifier_layer.state_dict(),
                              'feature_residual': feature_residual_layer.state_dict(),
                              'class_residual': class_residual_layer.state_dict()}

            log_str = 'iter: {:d}, all_accu: {:.4f},\ttime: {:.4f}'.format(num_iter, temp_acc, time.time() - since)
            config['logger'].logger.debug(log_str)
            config['results'][num_iter].append(temp_acc)

        # This has any effect only on modules such as Dropout or BatchNorm.
        base_network.train(True)
        classifier_layer.train(True)
        feature_residual_layer.train(True)
        class_residual_layer.train(True)

        # freeze BN layers
        for m in base_network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.training = False
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        # load data
        if num_iter % len_train_source == 0:
            iter_source = iter(dset_loaders['source'])
        if num_iter % len_train_target == 0:
            iter_target = iter(dset_loaders['target'])
        inputs_source, labels_source = iter_source.next()
        inputs_target, _ = iter_target.next()
        batch_size = len(labels_source)
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        optimizer = lr_scheduler(optimizer, num_iter / config['max_iter'], **schedule_param)
        optimizer.zero_grad()

        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features_base = base_network(inputs)
        features_residual = feature_residual_layer(features_base)
        total_feature_residual = features_base + features_residual

        # source classification loss with original features
        output_base = classifier_layer(features_base)
        classifier_loss = class_criterion(output_base[:batch_size, :], labels_source)

        # target residual feature entropy loss
        residual_output_base = classifier_layer(total_feature_residual)
        output_residual = class_residual_layer(residual_output_base)
        total_output_residual = residual_output_base + output_residual
        softmax_output_base = softmax_layer(output_base)
        total_softmax_residual = softmax_layer(total_output_residual)
        entropy_loss = EntropyLoss(total_softmax_residual[batch_size:, :])

        # alignment of L task-specific feature layers (Here, we have one layer)
        transfer_loss = MMD(features_base[:batch_size, :],
                            total_feature_residual[batch_size:, :])
        # alignment of softmax layer
        transfer_loss += MMD(softmax_output_base[:batch_size, :],
                             total_softmax_residual[batch_size:, :],
                             kernel_num=1, fix_sigma=1.68)

        source_labels_data = labels_source.data.float()
        sum_reg_loss = 0
        for k in range(class_num):
            source_k_index = []
            for index, source_k in enumerate(source_labels_data):
                # find all indexes of k-th class source samples
                if source_k == k:
                    source_k_index.append(index)
            fea_reg_loss = 0
            out_reg_loss = 0
            if len(source_k_index) > 0:
                # random subset indexes of source samples
                source_rand_index = []
                index = 0
                for z in range(batch_size):
                    prob = random.random()
                    if prob < config['random_prob'] / class_num:
                        source_rand_index.append(index)
                        index += 1

                if len(source_rand_index) > 0:
                    # source feature of k-th class
                    source_k_fea = features_base.index_select(0, torch.tensor(source_k_index, dtype=torch.long).cuda())
                    source_k_out = output_base.index_select(0, torch.tensor(source_k_index, dtype=torch.long).cuda())

                    # random selected source feature
                    source_rand_fea = total_feature_residual.index_select(0, torch.tensor(source_rand_index,
                                                                                          dtype=torch.long).cuda())
                    source_rand_out = total_output_residual.index_select(0, torch.tensor(source_rand_index,
                                                                                         dtype=torch.long).cuda())

                    fea_reg_loss = MMD_reg(source_k_fea, source_rand_fea)
                    out_reg_loss = MMD_reg(source_k_out, source_rand_out, kernel_num=1, fix_sigma=1.68)

            sum_reg_loss += (fea_reg_loss + out_reg_loss)

        total_loss = classifier_loss + \
                     config['loss']['alpha_off'] * (transfer_loss +
                                                    config['loss']['constant_off'] * sum_reg_loss) + \
                     config['loss']['beta_off'] * entropy_loss
        total_loss.backward()
        optimizer.step()

        if num_iter % config['val_iter'] == 0:
            config['logger'].logger.debug(
                'class: {:.4f}\tmmd: {:.4f}\tmmd_seg: {:.4f}\tentropy: {:.4f}'.format(classifier_loss.item(),
                                                                                      transfer_loss.item(),
                                                                                      config['loss'][
                                                                                          'constant_off'] * sum_reg_loss,
                                                                                      entropy_loss.item() *
                                                                                      config['loss']['beta_off']))
            if config['is_writer']:
                config['writer'].add_scalars('train', {'class': classifier_loss.item(), 'mmd': transfer_loss.item(),
                                                       'mmd_seg': config['loss']['constant_off'] * sum_reg_loss.item(),
                                                       'entropy': config['loss']['beta_off'] * entropy_loss.item()},
                                             num_iter)

    if config['is_writer']:
        config['writer'].close()

    torch.save(best_model, osp.join(config['path']['model'], config['task'] + '_best_model.pth'))
    return best_acc


def empty_dict(config):
    config['results'] = {}
    for i in range(config['max_iter'] // config['val_iter'] + 1):
        key = config['val_iter'] * i
        config['results'][key] = []
    config['results']['best'] = []


def print_dict(config):
    for i in range(config['max_iter'] // config['val_iter'] + 1):
        key = config['val_iter'] * i
        log_str = 'setting: {:d}, average: {:.4f}'.format(key, np.average(config['results'][key]))
        config['logger'].logger.debug(log_str)
    log_str = 'best, average: {:.4f}'.format(np.average(config['results']['best']))
    config['logger'].logger.debug(log_str)
    config['logger'].logger.debug('-' * 100)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generalized Domain Conditioned Adaptation Network')
    parser.add_argument('--seed', type=int, default=1, help='manual seed')
    parser.add_argument('--gpu', type=str, nargs='?', default='1', help='device id to run')
    parser.add_argument('--net', type=str, default='50', choices=['50', '101'])
    parser.add_argument('--data_set', default='home', choices=['home', 'domainnet', 'office', 'clef'], help='data set')
    parser.add_argument('--source_path', type=str, default='data/list/office/Art_65.txt', help='The source list')
    parser.add_argument('--target_path', type=str, default='data/list/office/Clipart_65.txt', help='The target list')
    parser.add_argument('--test_path', type=str, default='data/list/office/Clipart_65.txt', help='The test list')
    parser.add_argument('--output_path', type=str, default='snapshot/', help='save ``log/scalar/model`` file path')
    parser.add_argument('--task', type=str, default='ac', help='transfer task name')
    parser.add_argument('--max_iter', type=int, default=20001, help='max iterations')
    parser.add_argument('--val_iter', type=int, default=500, help='interval of two continuous test phase')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (Default:1e-4')
    parser.add_argument('--random_prob', type=float, default=0.8, help='the probability of random sampling')
    parser.add_argument('--batch_size', type=int, default=36, help='mini batch size')
    parser.add_argument('--beta_off', type=float, default=0.1, help='target entropy loss weight ')
    parser.add_argument('--alpha_off', type=float, default=1.5, help='discrepancy loss weight')
    parser.add_argument('--is_writer', action='store_true', help='If added to sh, record for tensorboard')
    args = parser.parse_args()

    # seed for everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONASHSEED'] = str(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = {'seed': args.seed, 'gpu': args.gpu, 'max_iter': args.max_iter, 'val_iter': args.val_iter,
              'random_prob': args.random_prob, 'data_set': args.data_set, 'task': args.task,
              'prep': {'test_10crop': True, 'params': {'resize_size': 256, 'crop_size': 224}},
              'network': {'name': args.net, 'params': {'resnet_name': args.net, 'class_num': 65}},
              'optimizer': {'type': optim.SGD,
                            'optim_params': {'lr': args.lr, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
                            'lr_type': 'inv', 'lr_param': {'lr': args.lr, 'gamma': 1.0, 'power': 0.75}},
              'data': {
                  'source': {'list_path': args.source_path, 'batch_size': args.batch_size},
                  'target': {'list_path': args.target_path, 'batch_size': args.batch_size},
                  'test': {'list_path': args.test_path, 'batch_size': args.batch_size}},
              'output_path': args.output_path + args.data_set,
              'path': {'log': args.output_path + args.data_set + '/log/',
                       'scalar': args.output_path + args.data_set + '/scalar/',
                       'model': args.output_path + args.data_set + '/model/'},
              'is_writer': args.is_writer
              }
    if config['data_set'] == 'home':
        config['network']['params']['class_num'] = 65
    elif config['data_set'] == 'domainnet':
        config['network']['params']['class_num'] = 345
    elif config['data_set'] == 'office':
        config['network']['params']['class_num'] = 31
    elif config['data_set'] == 'clef':
        config['network']['params']['class_num'] = 12
        config['optimizer']['lr_param']['gamma'] = 3e-4
    else:
        raise ValueError('dataset %s not found!' % (config['data_set']))

    config['loss'] = {'alpha_off': args.alpha_off,
                      'constant_off': 1 / config['network']['params']['class_num'],
                      'beta_off': args.beta_off}

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
        os.makedirs(config['path']['log'])
        os.makedirs(config['path']['scalar'])
        os.makedirs(config['path']['model'])
    if config['is_writer']:
        config['writer'] = SummaryWriter(log_dir=config['path']['scalar'])
    config['logger'] = Logger(logroot=config['path']['log'], filename=config['task'], level='debug')

    config['logger'].logger.debug(str(config))

    empty_dict(config)
    config['results']['best'].append(train(config))
    print_dict(config)
