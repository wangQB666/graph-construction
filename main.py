import argparse
from engine import *
from models import *
from util import *
from dataLoader import *
from transformers import BertModel

import datetime

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Training Super-parameters')

parser.add_argument('-seed', default=0, type=int, metavar='N',
                    help='random seed')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[40], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--G-lr', '--Generator-learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--D-lr', '--Discriminator-learning-rate', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--B-lr', '--Bert-learning-rate', default=0.001, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', #
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
parser.add_argument('--evaluate', default=False, type=bool,
                    help='evaluate')
parser.add_argument('--save_model_path', default='./checkpoint', type=str,
                    help='path to save checkpoint (default: none)')
parser.add_argument('--data_type', default='All', type=str,
                    help='The type of data')
parser.add_argument('--data_path', default='../datasets/AAPD/aapd2.csv', type=str,
                    help='path of data')
parser.add_argument('--bert_trainable', default=True, type=bool,
                    help='bert_trainable')
parser.add_argument('--use_previousData', default=0, type=int,
                    help='use_previousData')
parser.add_argument('--model_type', default='MABert', type=str,
                    help='The type of model to train')
parser.add_argument('--method', default='MultiLabelMAP', type=str,
                    help='Method')
parser.add_argument('--overlength_handle', default='truncation', type=str,
                    help='overlength_handle')
parser.add_argument('--min_tagFrequence', default=0, type=int,
                    help='min_tagFrequence')
parser.add_argument('--max_tagFrequence', default=100000, type=int,
                    help='max_tagFrequence')
parser.add_argument('--intanceNum_limit', default=2000, type=int,
                    help='max_tagFrequence')
parser.add_argument('--data_split', default=0.5, type=float,
                    help='data_split')
parser.add_argument('--experiment_no', default='01', type=str,
                    help='experiment_no')
parser.add_argument('--test_description', default='01', type=str,
                    help='test_description')

global args, use_gpu
args = parser.parse_args()

use_gpu = torch.cuda.is_available()

result_path = os.path.join('result', datetime.date.today().strftime('%Y%m%d'))
log_dir = os.path.join(result_path, 'logs')
if not os.path.exists(result_path):
    os.makedirs(result_path)
method_str = args.experiment_no + '_' + args.data_path.split("/")[-1]

# result_method_path = os.path.join(result_path, method_str)
# if not os.path.exists(result_method_path):
#     os.makedirs(result_method_path)

fo = open(os.path.join(result_path, method_str + '.txt'), "a+")
print('#' * 100 + '\n')
fo.write('#' * 100 + '\n')
setting_str = 'Setting: \t batch-size: {} \t epoch_step: {} \t G_LR: {} \t D_LR: {} \t B_LR: {}'\
              '\ndevice_ids: {} \t data_path: {} \t bert_trainable: {}' \
              '\nuse_previousData: {} \t method: {} \t overlength_handle: {} \t data_split: {} \n' \
              'experiment_no: {} \t test_description: {} \t model_type: {} \t evaluate: {}\n'.format(
                args.batch_size, args.epoch_step, args.G_lr, args.D_lr, args.B_lr,
                args.device_ids, args.data_path, args.bert_trainable,
                args.use_previousData, args.method, args.overlength_handle, args.data_split,
                args.experiment_no, args.test_description, args.model_type, args.evaluate)

print(setting_str)
fo.write(setting_str)

data_config = {'overlength_handle': args.overlength_handle, 'intanceNum_limit': args.intanceNum_limit,
               'min_tagFrequence': args.min_tagFrequence, 'max_tagFrequence': args.max_tagFrequence,
               'data_split': args.data_split, 'method': args.method, 'method_str': method_str}

dataset, encoded_tag, tag_mask = load_data(data_config=data_config,
                                           data_path=args.data_path,
                                           data_type=args.data_type,
                                           use_previousData=args.use_previousData)

data_size = "train_data_size: {} \nunlabeled_train_data: {} \nval_data_size: {} \ntag_num: {} \n".format(
    len(dataset.train_data), len(dataset.unlabeled_train_data), len(dataset.test_data), len(dataset.tag2id))
print(data_size)
fo.write(data_size)

state = {'batch_size': args.batch_size, 'max_epochs': args.epochs, 'evaluate': args.evaluate,
         'resume': args.resume, 'num_classes': dataset.get_tags_num(), 'difficult_examples': False,
         'save_model_path': args.save_model_path, 'log_dir': log_dir, 'workers': args.workers,
         'epoch_step': args.epoch_step, 'lr': args.D_lr, 'encoded_tag': encoded_tag, 'tag_mask': tag_mask,
         'device_ids': args.device_ids, 'print_freq': args.print_freq, 'id2tag': dataset.id2tag,
         'result_file': fo, 'method': args.method}

# if args.evaluate:
#     state['evaluate'] = True

bert = BertModel.from_pretrained('../datasets/bert-base-uncased')

# define loss function (criterion)
criterion = nn.BCELoss() #nn.MultiLabelSoftMarginLoss()#nn.CrossEntropyLoss()#

model = {}
optimizer = {}

model['Generator'] = Generator(bert, len(dataset.tag2id))
# define optimizer
optimizer['Generator'] = torch.optim.SGD([{'params': model['Generator'].parameters(), 'lr': args.G_lr}],
                                         momentum=args.momentum, weight_decay=args.weight_decay)

# optimizer['Generator'] = torch.optim.Adam(model['Generator'].get_config_optim(args.G_lr))

if args.model_type == 'MLPBert':
    model['Classifier'] = MLPBert(bert, num_classes=len(dataset.tag2id), hidden_dim=512, hidden_layer_num=1, bert_trainable=True)

    optimizer['Classifier'] = torch.optim.SGD(model['Classifier'].get_config_optim(args.D_lr, args.B_lr),
                                              momentum=args.momentum, weight_decay=args.weight_decay)

    engine = MultiLabelMAPEngine(state)

elif args.model_type == 'MABert':

    model['Classifier'] = MABert(bert, num_classes=len(dataset.tag2id), bert_trainable=args.bert_trainable, device=args.device_ids[0])

    optimizer['Classifier'] = torch.optim.SGD(model['Classifier'].get_config_optim(args.D_lr, args.B_lr),
                                momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer['Classifier'] = torch.optim.Adam(model['Classifier'].get_config_optim(args.D_lr, args.B_lr))

    if args.method == 'MultiLabelMAP':
        engine = MultiLabelMAPEngine(state)
    elif args.method == 'semiGAN_MultiLabelMAP':
        engine = semiGAN_MultiLabelMAPEngine(state)

engine.learning(model, criterion, dataset, optimizer)

fo.close()