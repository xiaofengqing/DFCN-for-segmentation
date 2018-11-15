import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from DFCN.models import get_model
from data_loader import get_loader, get_data_path
from loss import cross_entropy2d
from metrics import scores
from lr_scheduling import *

def train(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))
    n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)
    model_path = '/home/models/'
    # Setup visdom for visualization
    # vis = visdom.Visdom()
    #
    # loss_window = vis.line(X=torch.zeros((1,)).cpu(),
    #                        Y=torch.zeros((1)).cpu(),
    #                        opts=dict(xlabel='minibatches',
    #                                  ylabel='Loss',
    #                                  title='Training Loss',
    #                                  legend=['Loss']))

    # Setup Model
    model = get_model(args.arch, n_classes)

    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        # test_image, test_segmap = loader[0]
        # test_image = Variable(test_image.unsqueeze(0).cuda())
    # else:
    #     test_image, test_segmap = loader[0]
    #     test_image = Variable(test_image.unsqueeze(0))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    for epoch in range(args.n_epoch):
        loss_sum = 0
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            iter = len(trainloader)*epoch + i
            poly_lr_scheduler(optimizer, args.l_rate, iter)
            
            optimizer.zero_grad()
            outputs = model(images)

            loss = cross_entropy2d(outputs, labels)

            loss.backward()
            optimizer.step()

            # vis.line(
            #     X=torch.ones((1, 1)).cpu() * i,
            #     Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
            #     win=loss_window,
            #     update='append')

            loss_sum = loss_sum+loss.data[0]
            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))
        mean_loss = loss_sum/len(trainloader)
        print("Epoch [%d/%d] mean_Loss: %.4f" % (epoch + 1, args.n_epoch, mean_loss))

        # test_output = model(test_image)
        # predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        # target = loader.decode_segmap(test_segmap.numpy())

        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))

        # torch.save(model,  model_path+"{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))
    torch.save(model, model_path + "{}_{}_{}_{}_{}_v.pkl".format(args.arch, args.dataset, args.batch_size,args.l_rate,args.n_epoch))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s',
                        help='Architecture to use [\'fcn8s, unet, segnet, DFCN etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='Potsdam',
                        help='Dataset to use [\'pascal, Vaihingen, Potsdam etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=50,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    args = parser.parse_args()
    train(args)
