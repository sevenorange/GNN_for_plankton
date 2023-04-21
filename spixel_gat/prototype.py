from tqdm import tqdm
import fire

import os
import copy
import time

import numpy as np
import scipy as sp
import scipy.io
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

from model import GAT_MNIST
import util

import matplotlib.pyplot as plt
from skimage import io, transform

to_cuda = util.to_cuda

def images_from_matlab_to_numpy(images):
    return images.transpose(3,0,1,2).astype(np.float64)/256.0
def labels_from_matlab_to_numpy(labels):
    return labels.squeeze()

def process_dataset(dset_folder,subset="train"):
    if subset not in ["train","test","extra"]:
        raise ValueError("Subset must be one of ('train', 'test', 'extra')")
    prefix = "{dset_folder}/".format(dset_folder=dset_folder)
    subset_prefix = "./gnn_datasets/"+"slic_100/"+"{subset}_".format(subset=subset)
    print(subset_prefix)
    try:
        labels = np.load(subset_prefix + "labels.npy")
        graphs = []
        for i in range(len(labels)):
            g = (
                np.load(subset_prefix + "{}_h.npy".format(i)),
                np.load(subset_prefix + "{}_e.npy".format(i))
            )
            graphs.append(g)
        graphs = np.array(graphs)
    except IOError:
        print("Couldn't find the processed graph dataset, processing it from scratch")
        # 加载原数据集
        dataset_label = []
        # graphs = []
        dirname = dset_folder + "/" + subset
        count = 0
        print("Processing images into graphs...", end="")
        ptime = time.time()
        for i, dir in enumerate(os.listdir(dirname)):
            label = i
            subdir = dirname + "/" + dir
            for img in os.listdir(subdir):
                _, postfix = os.path.splitext(img)
                if postfix not in ['.jpg', '.png', '.jpeg','.JPG', '.PNG', '.JPEG', '.bmp']:
                    continue
                a = plt.imread(subdir + "/" + img)                
                a = transform.resize(a, (224, 224))
                a = a.astype(np.float64)/256.0
                # print(a.shape)
                h, e = util.get_graph_from_image(a)
                np.save(subset_prefix + "{}_h.npy".format(count), h)
                np.save(subset_prefix + "{}_e.npy".format(count), e)
                # graphs.append([h, e])
                dataset_label.append(i)
                count += 1
                print(count)
        # dataset_img = np.array(graphs)
        dataset_label = np.array(dataset_label)
        # print(dataset_label.shape)  
        # imgs = images_from_matlab_to_numpy(dataset_img)
        labels = labels_from_matlab_to_numpy(dataset_label)    
        np.save(subset_prefix + "labels.npy", labels)
        print("lable classes: ", len(np.unique(labels)))
        ptime = time.time() - ptime
        print(" Took {ptime}s".format(ptime=ptime))
        
    labels = labels.astype(util.NP_TORCH_LONG_DTYPE)
    return graphs, labels

def train_model(
        epochs,
        batch_size,
        use_cuda,
        dset_folder,
        disable_tqdm=False,
        ):
    print("Reading dataset... ", end="")
    ptime = time.time()
    graphs, labels = process_dataset(dset_folder,"train")
    print(len(np.unique(labels)))
    ptime = time.time() - ptime
    print(" Took {ptime}s".format(ptime=ptime))
    train_idx, valid_idx = map(np.array,util.split_dataset(labels))
    
    model_args = []
    model_kwargs = {}
    model = GAT_MNIST(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    if use_cuda:
        model = model.cuda()
    
    opt = torch.optim.Adam(model.parameters(), lr = 0.1)
    
    best_valid_acc = 0.
    best_model = copy.deepcopy(model)
    
    last_epoch_train_loss = 0.
    last_epoch_train_acc = 0.
    last_epoch_valid_acc = 0.
    
    valid_log_file = open("log.valid", "w")
    interrupted = False
    for e in tqdm(range(epochs), total=epochs, desc="Epoch ", disable=disable_tqdm,):
        try:
            train_losses, train_accs = util.train(model, opt, graphs, labels, train_idx, batch_size=batch_size, use_cuda=use_cuda, disable_tqdm=disable_tqdm,)
            
            last_epoch_train_loss = np.mean(train_losses)
            last_epoch_train_acc = 100*np.mean(train_accs)
        except KeyboardInterrupt:
            print("Training interrupted!")
            interrupted = True
        
        valid_accs = util.test(model,graphs,labels,valid_idx,use_cuda,desc="Validation ", disable_tqdm=disable_tqdm,)
                
        last_epoch_valid_acc = 100*np.mean(valid_accs)
        
        if last_epoch_valid_acc>best_valid_acc:
            best_valid_acc = last_epoch_valid_acc
            best_model = copy.deepcopy(model)
        
        tqdm.write("EPOCH SUMMARY {loss:.4f} {t_acc:.2f}% {v_acc:.2f}%".format(loss=last_epoch_train_loss, t_acc=last_epoch_train_acc, v_acc=last_epoch_valid_acc))
        tqdm.write("EPOCH SUMMARY {loss:.4f} {t_acc:.2f}% {v_acc:.2f}%".format(loss=last_epoch_train_loss, t_acc=last_epoch_train_acc, v_acc=last_epoch_valid_acc), file=valid_log_file)
        
        if interrupted:
            break
    model_path = './gnn_datasets/slic_100/save_models/'
    util.save_model(model_path + "best",best_model)
    util.save_model(model_path + "last",model)


def test_model(
        use_cuda,
        dset_folder,
        disable_tqdm=False,
        ):
    best_model = GAT_MNIST(num_features=util.NUM_FEATURES, num_classes=util.NUM_CLASSES)
    model_path = './gnn_datasets/slic_100/save_models/'
    util.load_model(model_path + "best",best_model)
    if use_cuda:
        best_model = best_model.cuda()
    
    test_graphs, test_labels = process_dataset(dset_folder,"test")
    
    test_accs = util.test(best_model, test_graphs, test_labels, list(range(len(test_labels))), use_cuda, desc="Test ", disable_tqdm=disable_tqdm,)
    test_acc = 100*np.mean(test_accs)
    print("TEST RESULTS: {acc:.2f}%".format(acc=test_acc))

def main(
        dataset:bool=True,
        train:bool=True,
        test:bool=True,
        epochs:int=300,
        batch_size:int=128,
        use_cuda:bool=True,
        disable_tqdm:bool=False,
        dset_folder:str = "./Crustacea" # 数据集路径
        ):
    use_cuda = use_cuda and torch.cuda.is_available()
    if dataset:
        # TODO? Download
        process_dataset(dset_folder,"train")
        process_dataset(dset_folder,"test")
    if train:
        train_model(
                epochs = epochs,
                batch_size = batch_size,
                use_cuda = use_cuda,
                dset_folder = dset_folder,
                disable_tqdm = disable_tqdm,
                )
    if test:
        test_model(
                use_cuda=use_cuda,
                dset_folder = dset_folder,
                disable_tqdm = disable_tqdm,
                )

if __name__ == "__main__":
    fire.Fire(main)
