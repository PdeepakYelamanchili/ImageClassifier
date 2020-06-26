# Imports python modules
import argparse


def get_predict_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type = str, default = './flowers/test/1/image_06743.jpg', help = 'Dataset path')
    parser.add_argument('--checkpoint_path', type = str, default = './checkpoint.pth', help = 'load checkpoint')
    parser.add_argument('--top_k', type = int, default = 5, help = 'Top K most likely classes')
    parser.add_argument('--gpu', action = "store_true", default = True, help = 'Use GPU if available')

    args = parser.parse_args()

    return args.image_path, args.checkpoint_path, args.top_k, args.gpu