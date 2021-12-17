import numpy as np
import time
import argparse
import random
from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa, "Select either 'lsa' or 'dsa'"
    print(args)
    

    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        # print(x_test.shape)
        # Load pre-trained model.
        # model = load_model("/content/drive/MyDrive/sadl11/model/model_mnist_LeNet5.h5")
        # model.summary()
        model= load_model("/content/drive/MyDrive/sadl11/model/model_mnist_LeNet1.h5")
        # # You can select some layers you want to test.
        # LeNet1
        layer_names = ["conv2d_1"]
        #LeNet5
        # layer_names = ["activation_13"]

        # # Load target set.
        # # x_target = np.load("./adv/adv_mnist_{}.npy".format(args.target))

    elif args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        model = load_model("/content/drive/MyDrive/sadl11/model/model_cifar.h5")
        
        # # model.summary()

        # # layer_names = [
        # #     layer.name
        # #     for layer in model.layers
        # #     if ("activation" in layer.name or "pool" in layer.name)
        # #     and "activation_9" not in layer.name
        # # ]
        layer_names = ["activation_3"]

        # x_target = np.load("./adv/adv_cifar_{}.npy".format(args.target))
    # # for i in range(500):
    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    # x_test = x_test.astype("float32")
    # x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)
    # print("x_test",np.array(x_test).shape)
    # # x_testLSC=np.load("/content/drive/MyDrive/sadl11/tmp/x_tcovlsc.npy")
    # # x_testDSC=np.load("/content/drive/MyDrive/sadl11/tmp/x_tcovdsc.npy")
    if args.lsa:
        x_testLSC=np.load("/content/drive/MyDrive/sadl11/tmp/x_tcovlsc.npy")
        test_lsa = fetch_lsa(model, x_train, x_testLSC, "test", layer_names, args)
        test_cov1 = get_sc(-140, args.upper_bound, args.n_bucket, test_lsa) 
        np.save("/content/drive/MyDrive/sadl11/tmp/test_cov.npy",test_cov1)
        print("args.upper_bound, args.n_bucket", args.upper_bound, args.n_bucket)
        print(infog("{} LSC coverage: ".format("test") + str(test_cov1)))
        # target_lsa = fetch_lsa(model, x_train, x_target, args.target, layer_names, args)
        # target_cov = get_sc(
            # 0, args.upper_bound, args.n_bucket, test_lsa + target_lsa[:1000])

        # auc = compute_roc_auc(test_lsa)
        # print(infog("ROC-AUC: "+ str(auc * 100)))

        # test_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)
        # test_cov = get_sc(-140, args.upper_bound, args.n_bucket, test_lsa)

        # target_lsa = fetch_lsa(model, x_train, x_target, args.target, layer_names, args)
        # target_cov = get_sc(
        #     -140, args.upper_bound, args.n_bucket, test_lsa + target_lsa[:1000]
        # )

        # auc = compute_roc_auc(test_lsa, target_lsa)
        # print(infog("ROC-AUC: " + str(auc * 100)))
    if args.dsa:
        x_testDSC=np.load("/content/drive/MyDrive/sadl11/tmp/x_tcovdsc.npy")
        test_dsa = fetch_dsa(model, x_train, x_testDSC, "test", layer_names, args)
        test_cov = get_sc(0, args.upper_bound, args.n_bucket, test_dsa) 
        np.save("/content/drive/MyDrive/sadl11/tmp/DSC_cov.npy",test_cov)
        print(infog("{} DSC coverage: ".format("test") + str(test_cov)))
        # target_dsa = fetch_dsa(model, x_train, x_target, args.target, layer_names, args)
        # target_cov = get_sc(
        #     np.amin(target_dsa), args.upper_bound, args.n_bucket, target_dsa
        # )

    #     auc = compute_roc_auc(test_dsa, target_dsa)
    #     print(infog("ROC-AUC: " + str(auc * 100)))    
    # print(infog("{} coverage: ".format("test") + str(test_cov)))
    # print(infog("{} coverage: ".format("test + " + args.target) + str(target_cov)))
