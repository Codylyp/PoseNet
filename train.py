import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from models.PoseNet import PoseNet, PoseLoss
from data.DataSource import *
from optparse import OptionParser
import os
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(epochs, batch_size, learning_rate, save_freq, data_dir):
    # train dataset and train loader
    datasource = DataSource(data_dir, train=True)
    train_loader = Data.DataLoader(dataset=datasource, batch_size=batch_size, shuffle=True)

    # load model
    posenet = PoseNet().to(device)

    # loss function
    criterion = PoseLoss(0.3, 0.3, 1., 300, 300, 300)

    # train the network
    optimizer = torch.optim.Adam(nn.ParameterList(posenet.parameters()),
                     lr=learning_rate, eps=1,
                     weight_decay=0.0625,
                     betas=(0.9, 0.999))

    #optimizer = torch.optim.SGD(nn.ParameterList(posenet.parameters()),
                  #   lr=learning_rate, momentum=0.0009,
                  #   weight_decay=0.0005)

    loss_sum = 0.0
    loss_total = []
    batches_per_epoch = len(train_loader.batch_sampler)
    for epoch in range(epochs):
        print("Starting epoch {}:".format(epoch))
        posenet.train()
        for step, (images, poses) in enumerate(train_loader):
            b_images = Variable(images, requires_grad=True).to(device)
            b_images = b_images.type(torch.cuda.FloatTensor)###############
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
        #torch.save(posenet.state_dict(), save_path)
        #print("Network saved!")

            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(poses), requires_grad=True).to(device)
            #b_poses = Variable(torch.tensor(poses,dtype=torch.float32), requires_grad=True).to(device)
            #b_poses = b_poses.type(torch.FloatTensor)

            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
            ########################
            #mean = [0.5,0.5,0.5]
            #std = [0.5,0.5,0.5]
            #p1_x = unnormalize(p1_x,mean,std)
            ###########################
            #b_poses = b_poses/100
            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("{}/{}: loss = {}".format(step+1, batches_per_epoch, loss))
            loss_sum += loss.item()########################
        loss_total.append(loss_sum/batches_per_epoch)#############
        loss_sum = 0.0###################

        # Save state
        if epoch % save_freq == 0:
            save_filename = 'epoch_{}.pth'.format(str(epoch+1).zfill(5))
            save_path = os.path.join('checkpoints', save_filename)
            torch.save(posenet.state_dict(), save_path)

    ###############################################################
    plt.figure(figsize=(10, 5))
    plt.plot(loss_total, color='b', label='Training loss')
    plt.scatter(np.arange(len(loss_total)),loss_total, color='b')
    plt.legend()
    plt.title("The trend of training loss during 21 epochs")
    plt.xticks(np.arange(len(loss_total)))
    plt.show()
    plt.savefig('train_loss.png')
    ######################################################################

def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', default=21, type='int')
    parser.add_option('--learning_rate', default=0.0001)
    parser.add_option('--batch_size', default=20, type='int')
    parser.add_option('--save_freq', default=20, type='int')
    parser.add_option('--data_dir', default='data/datasets/KingsCollege/')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    main(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, save_freq=args.save_freq, data_dir=args.data_dir)
