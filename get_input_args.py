
# Imports python modules
import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type = str, default = 'vgg16',  help = 'Model')
    parser.add_argument('--learn_rate', type = float, default = 0.001, help = 'Learn rate')
    parser.add_argument('--hidden_units', type = int, default = 1024, help = 'hidden units')
    parser.add_argument('--epochs', type = int, default = 5, help = 'epochs')
    parser.add_argument('--gpu', action = "store_true", default = True, help = 'GPU')

    args = parser.parse_args()

    return args.network, args.learn_rate, args.hidden_units, args.epochs, args.gpu