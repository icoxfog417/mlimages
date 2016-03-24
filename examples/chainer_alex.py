import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import argparse
import numpy as np
import chainer
from chainer import optimizers
from chainer import serializers
import alex
from mlimages.gather.imagenet import ImagenetAPI
from mlimages.label import LabelingMachine
from mlimages.training import TrainingData
from mlimages.model import ImageProperty


DATA_DIR = os.path.join(os.path.dirname(__file__), "./data/imagenet/")
IMAGES_ROOT = os.path.join(DATA_DIR, "./images")

LABEL_FILE = os.path.join(os.path.dirname(__file__), "./data/imagenet/label.txt")
LABEL_DEF_FILE = os.path.join(os.path.dirname(__file__), "./data/imagenet/label_def.txt")

MEAN_IMAGE_FILE = os.path.join(os.path.dirname(__file__), "./data/imagenet/mean_image.png")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "./data/imagenet/chainer_alex.model")
IMAGE_PROP = ImageProperty(width=227, resize_by_downscale=True)


def download_imagenet(wnid, limit=-1):
    api = ImagenetAPI(data_root=DATA_DIR, limit=limit, debug=True)
    api.logger.info("start to gather the ImageNet images.")
    folders = api.gather(wnid, include_subset=True)

    # rename images root folder
    images_root = os.path.join(DATA_DIR, folders[0])
    os.rename(images_root, IMAGES_ROOT)
    print("Down load has done.")


def make_label():
    machine = LabelingMachine(data_root=IMAGES_ROOT)
    lf = machine.label_dir_auto(label_file=LABEL_FILE, label_def_file=LABEL_DEF_FILE)


def show(limit, shuffle=True):
    td = TrainingData(LABEL_FILE, img_root=IMAGES_ROOT, image_property=IMAGE_PROP)
    _limit = limit if limit > 0 else 5
    iterator = td.fetch()
    if shuffle:
        import random
        shuffled = list(iterator)
        random.shuffle(shuffled)
        iterator = iter(shuffled)

    for i, im in enumerate(iterator):
        im.image.show()
        if i >= _limit - 1:
            break


def train(epoch=10, batch_size=32):
    td = TrainingData(LABEL_FILE, img_root=IMAGES_ROOT, image_property=IMAGE_PROP)

    # make mean image
    if not os.path.isfile(MEAN_IMAGE_FILE):
        print("make mean image...")
        td.make_mean_image(MEAN_IMAGE_FILE)
    else:
        td.mean_image_file = MEAN_IMAGE_FILE

    # train model
    #  setup
    print("begin training the model.")
    label_def = LabelingMachine.read_label_def(LABEL_DEF_FILE)
    model = alex.Alex(len(label_def))
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    epoch = epoch
    batch_size = batch_size

    for i in range(epoch):
        print("epoch_{0}: (learning rate={1})".format(i, optimizer.lr))
        td.shuffle(overwrite=True)

        for x_batch, y_batch in td.generate_batches(batch_size):
            x = chainer.Variable(np.asarray(x_batch))
            t = chainer.Variable(np.asarray(y_batch))

            optimizer.update(model, x, t)
            print("loss: {0}, accuracy: {1}".format(float(model.loss.data), float(model.accuracy.data)))

        serializers.save_npz(MODEL_FILE, model)
        optimizer.lr *= 0.97


def predict(limit):
    _limit = limit if limit > 0 else 5

    td = TrainingData(LABEL_FILE, img_root=IMAGES_ROOT, mean_image_file=MEAN_IMAGE_FILE, image_property=IMAGE_PROP)
    label_def = LabelingMachine.read_label_def(LABEL_DEF_FILE)
    model = alex.Alex(len(label_def))
    serializers.load_npz(MODEL_FILE, model)

    i = 0
    for arr, label in td.generate():
        x = np.ndarray((1,) + arr.shape, arr.dtype)
        x[0] = arr
        x = chainer.Variable(np.asarray(x), volatile="on")
        y = model.predict(x)
        p = np.argmax(y.data)
        im = td.data_to_image(arr)
        print("predict {0}, actual {1}".format(label_def[p], label_def[label]))
        im.image.show()
        i += 1
        if i >= _limit:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of Imagenet x AlexNet")
    parser.add_argument("task", type=str, help="task of script. " + "".join([
        "g: gather images", "l: make label file", "s: show training images (shuffle data when 'ss')",
        "t: train model", "p: predict"
    ]))
    parser.add_argument("-wnid", type=str, help="imagenet id (default is cats(n02121808))", default="n02121808")
    parser.add_argument("-limit", type=int, help="g: download image limit, s,p: show/predict image limit", default=-1)

    args = parser.parse_args()

    if args.task == "g":
        download_imagenet(args.wnid, args.limit)
    elif args.task == "l":
        print("create label data automatically.")
        make_label()
    elif args.task == "s":
        show(args.limit, shuffle=False)
    elif args.task == "ss":
        show(args.limit, shuffle=True)
    elif args.task == "t":
        train()
    elif args.task == "p":
        predict(args.limit)
