import mxnet as mx
import numpy as np
import os, time, logging, math, argparse
from tqdm import tqdm
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from time import sleep
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

# def parse_args():
#     parser = argparse.ArgumentParser(description='Gluon for FashionAI Competition',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--task', required=True, type=str,
#                         help='name of the classification task')
#     parser.add_argument('--model', required=True, type=str,
#                         help='name of the pretrained model from model zoo.')
#     parser.add_argument('-j', '--workers', dest='num_workers', default=2, type=int,
#                         help='number of preprocessing workers')
#     parser.add_argument('--num-gpus', default=1, type=int,
#                         help='number of gpus to use, 0 indicates cpu only')
#     parser.add_argument('--epochs', default=36, type=int,
#                         help='number of training epochs')
#     parser.add_argument('-b', '--batch-size', default=26   , type=int,
#                         help='mini-batch size')
#     parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
#                         help='initial learning rate')
#     parser.add_argument('--momentum', default=0.9, type=float,
#                         help='momentum')
#     parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-5, type=float,
#                         help='weight decay (default: 1e-4)')
#     parser.add_argument('--lr-factor', default=0.75, type=float,
#                         help='learning rate decay ratio')
#     parser.add_argument('--lr-steps', default='8,13,17,21,25,30', type=str,
#                         help='list of learning rate decay epochs as in str')
#     args = parser.parse_args()
#     return args

def calculate_ap(labels, outputs):
    cnt = 0
    ap = 0.
    for label, output in zip(labels, outputs):
        for lb, op in zip(label.asnumpy().astype(np.int),
                          output.asnumpy()):
            op_argsort = np.argsort(op)[::-1]
            lb_int = int(lb)
            ap += 1.0 / (1+list(op_argsort).index(lb_int))
            cnt += 1
    return ((ap, cnt))

def ten_crop(img, size):
    H, W = size
    iH, iW = img.shape[1:3]

    if iH < H or iW < W:
        raise ValueError('image size is smaller than crop size')##################

    img_flip = img[:, :, ::-1]
    crops = nd.stack(
        img[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img[:, 0:H, 0:W],
        img[:, iH - H:iH, 0:W],
        img[:, 0:H, iW - W:iW],
        img[:, iH - H:iH, iW - W:iW],

        img_flip[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
        img_flip[:, 0:H, 0:W],
        img_flip[:, iH - H:iH, 0:W],
        img_flip[:, 0:H, iW - W:iW],
        img_flip[:, iH - H:iH, iW - W:iW],
    )
    return (crops)




def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 299, 299), resize=320,
                                    rand_crop=True, rand_mirror=True,
                                    mean = np.array([0.485, 0.456, 0.406]),
                                    std = np.array([0.229, 0.224, 0.225]),
                                    brightness=0.125, contrast=0.125,
                                    saturation = 0.125, pca_noise = 0.05, inter_method = 10)

    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar())



def transform_val(data, label):
    im = data.astype('float32') / 255
    im = image.resize_short(im, 320)
    im, _ = image.center_crop(im, (299, 299))
    im = nd.transpose(im, (2,0,1))
    im = mx.nd.image.normalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return (im, nd.array([label]).asscalar())


def progressbar(i, n, bar_len=40):
    percents = math.ceil(100.0 * i / float(n))
    filled_len = int(round(bar_len * i / float(n)))
    prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('%s' % (percents))

def validate(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    AP = 0.
    AP_cnt = 0
    val_loss = 0
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]#
        metric.update(label, outputs)
        loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
        ap, cnt = calculate_ap(label, outputs)
        AP += ap
        AP_cnt += cnt
    _, val_acc = metric.get()
    return ((val_acc, AP / AP_cnt, val_loss / len(val_data)))

def train(task,model_name,dataset):
    logging.info('Start Training for Task: %s\n' % (task))

    #Initialize the net with pretrained model
    pretrained_net = gluon.model_zoo.vision.get_model(model_name, pretrained=True,prefix='inceptionv3_')
    finetune_net = gluon.model_zoo.vision.get_model(model_name, classes=task_list[task],prefix='inceptionv3_')
    finetune_net.features = pretrained_net.features
    finetune_net.features.lr_mult=0
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    # 用自己训练的多任务参数来初始化
    # pretrained_net = gluon.model_zoo.vision.get_model(model_name, #classes=task_num_class,
    #                                                   pretrained=True, prefix='inceptionv3_', ctx=ctx
    #                                                   )
    # finetune_net = gluon.model_zoo.vision.get_model(model_name, classes=task_list[task],
    #                                                 prefix='inceptionv3_', ctx=ctx
    #                                                 )
    # finetune_net.features = pretrained_net.features
    # finetune_net.output.initialize(init.Xavier(), ctx=ctx)
    # # if(model_name== 'densenet161'):
    # finetune_net.load_params('models/neckline_design_labels-resnet101_v2-30-features.params', ctx=ctx, allow_missing=True)
    # # if (model_name ==  'densenet201'or model_name ==  'resnet101_v2' ):
    # #     finetune_net.collect_params().reset_ctx(ctx)
    # finetune_net.hybridize()

    # Define DataLoader
    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(
            os.path.join(dataset, task, 'train'),
            transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='discard')

    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(
            os.path.join(dataset, task, 'val'),
            transform=transform_val),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    # Define Trainer
    trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
       'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    # trainer = gluon.Trainer(finetune_net.collect_params(), 'adam', {
    #     'learning_rate': lr, 'wd': 1e-4})
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    lr_counter = 0
    num_batch = len(train_data)

    # Start Training
    for epoch in range(epochs):
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*lr_factor)
            lr_counter += 1
        #if (epoch!=0 and epoch % 10 == 0):


        tic = time.time()
        train_loss = 0
        metric.reset()
        AP = 0.
        AP_cnt = 0

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with ag.record():
                outputs = [finetune_net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            metric.update(label, outputs)
            ap, cnt = calculate_ap(label, outputs)
            AP += ap
            AP_cnt += cnt

            # progressbar(i, num_batch-1)
            # percents = math.ceil(100.0 * i / float(n))
            # print('%s' % (math.ceil(100.0 * i / float(num_batch-1))))

        train_map = AP / AP_cnt
        _, train_acc = metric.get()
        train_loss /= num_batch

        val_acc, val_map, val_loss = validate(finetune_net, val_data, ctx)

        if (epoch !=0 and epoch%5==0):

            finetune_net.save_params('models/%s-%s-%s-overall.params' % (task, model_name, epoch))
            finetune_net.features.save_params('models/%s-%s-%s-features.params'%(task,model_name,epoch))


        # logging.info('[Epoch %d] Train-acc: %.3f, mAP: %.3f, loss: %.3f | Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1f' %
        #          (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, time.time() - tic))
        logging.info('[Epoch %d] Train-acc: %.3f, mAP: %.3f, loss: %.3f | Val-acc: %.3f,  mAP: %.3f, loss: %.3f | LR: %.6f,time: %.1f' %
            (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, trainer.learning_rate * lr_factor,time.time() - tic))

    logging.info('\n')
    return (finetune_net)


# Preparation
# args = parse_args()

task_list = {
    'collar_design_labels': 5,
    'skirt_length_labels': 6,
    'lapel_design_labels': 5,
    'neckline_design_labels': 10,
    'coat_length_labels': 8,
    'neck_design_labels': 5,
    'pant_length_labels': 6,
    'sleeve_length_labels': 9
}

# task_num_class = task_list[task]

# model_name = args.model

epochs = 31
lr = 0.0003
batch_size = 24
momentum = 0.9
wd = 1e-5

lr_factor = 0.75
lr_steps = [5,10,15] + [np.inf]
# lr_steps = [10] + [np.inf]

num_gpus = 2
num_workers = 2
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = batch_size * max(num_gpus, 1)

logging.basicConfig(level=logging.INFO,
                    handlers = [
                        logging.StreamHandler(),
                        logging.FileHandler('training.log')
                    ])


if __name__ == "__main__":
    # net = train('pant_length_labels','inceptionv3')

    # net = train('skirt_length_labels', 'densenet201', 'data/train_valid_densenet')
    # net = train('skirt_length_labels', 'resnet152_v2', 'data/train_valid_resnet')
    # net = train('pant_length_labels', 'densenet161', 'data/train_valid')


    # net = train('lapel_design_labels', 'resnet152_v2', 'data/train_valid_resnet')
    # net = train('neckline_design_labels', 'densenet161', 'data/train_valid_densenet')
    # net = train('coat_length_labels', 'densenet161', 'data/train_valid')
    # net = train('neck_design_labels', 'densenet201', 'data/train_valid_densenet')

    # net = train('coat_length_labels', 'resnet101_v2', 'data/train_valid_resnet')

    # net = train('pant_length_labels', 'resnet101_v2', 'data/train_valid_resnet')

    net = train('neckline_design_labels', 'resnet101_v2', 'data/train_valid_resnet')

    # net = train('collar_design_labels', 'resnet101_v2', 'data/train_valid_resnet')



