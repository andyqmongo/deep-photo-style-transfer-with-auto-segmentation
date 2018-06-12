from __future__ import print_function

import argparse
import os
import sys
import time
from PIL import Image
import tensorflow as tf
import numpy as np
from scipy import misc

from model import FCN8s, PSPNet50, ENet, ICNet

save_dir = './output/'
model_path = {'pspnet': './model/pspnet50.npy',
              'fcn': './model/fcn.npy',
              'enet': './model/cityscapes/enet.ckpt',
              'icnet': './model/cityscapes/icnet.npy'}

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.",
                        required=True)
    parser.add_argument("--save-dir", type=str, default=save_dir,
                        help="Path to save output.")
    parser.add_argument("--model", type=str, default='',
                        help="pspnet or fcn",
                        choices=['pspnet', 'fcn', 'enet', 'icnet'],
                        required=True)
    parser.add_argument("--isStyle", type=bool, default=True, help="whether the input image is style image or not" )
    parser.add_argument("--style_color", type=str, default="111111110")
    return parser.parse_args()

def printColorUsed(color):

    #color = color.reshape(-1,3)
    #black, red, green, blue, purple, yellow, lightblue, white
    answ = list("111111110")
    n = [0,0,0,0,0,0,0,0]
    n[0] = np.count_nonzero((color == [0, 0, 0]).all(axis = 2))
    n[1] = np.count_nonzero((color == [255, 0, 0]).all(axis = 2))
    n[2] = np.count_nonzero((color == [0, 255, 0]).all(axis = 2))
    n[3] = np.count_nonzero((color == [0, 0, 255]).all(axis = 2))
    n[4] = np.count_nonzero((color == [255, 0, 255]).all(axis = 2))
    n[5] = np.count_nonzero((color == [255, 255, 0]).all(axis = 2))
    n[6] = np.count_nonzero((color == [0, 255, 255]).all(axis = 2))
    n[7] = np.count_nonzero((color == [255, 255, 255]).all(axis = 2))
    maxColor = 0
    for i in np.arange(8):
        if n[i] <= 0:
            answ[i] = "0"
        if n[i] > maxColor:
            maxColor = n[i]
            answ[8] = str(i)
    print("".join(answ))

def main():
    args = get_arguments()

        
    if args.model == 'pspnet':
        model = PSPNet50(style_color=args.style_color)
    elif args.model == 'fcn':
        model = FCN8s()
    elif args.model == 'enet':
        model = ENet()
    elif args.model == 'icnet':
        model = ICNet()
        
    model.read_input(args.img_path)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    model.load(model_path[args.model], sess)

    preds = model.forward(sess)

    if args.isStyle:
        printColorUsed(preds[0])
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    misc.imsave(args.save_dir + args.model + '_' + model.img_name, preds[0])
    
if __name__ == '__main__':
    main()
