import matplotlib.pyplot as plt
import numpy
import pickle
import argparse
import os

def main(args):

    with open(os.path.join(args.hist_dir, "hist_stage_1.pickle"), "rb") as f:
        history1 = pickle.load(f)

    with open(os.path.join(args.hist_dir, "hist_stage_2.pickle"), "rb") as f:
        history2 = pickle.load(f)

    loss = history1['loss'] + history2['loss']
    val_loss = history1['val_loss'] + history2['val_loss']

    for i in range(len(loss)):
        loss[i] = float(loss[i])
    
    for i in range(len(val_loss)):
        val_loss[i] = float(val_loss[i])

    # Skip the loss for first 2 epochs
    plt.plot(loss[2:])
    plt.plot(val_loss[2:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise loss")
    # filenames
    parser.add_argument("-d", "--hist_dir", type = str, required = True, help = "Directory to history files")
    args = parser.parse_args()

    main(args)