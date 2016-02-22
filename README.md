# mlimages

gather and create image dataset for machine learning.

## Install

```
pip install mlimages
```

**This tool dependes on Python 3.5**

(but it works quickly because of async/await feature)


## Gather Images

### Imagenet

Confirm the **WordnetID** on the [ImageNet site](http://image-net.org/synset)

![imagenet](./doc/imagenet.PNG)

Then download it.

```
python gather.py -imagenet --wnid n11531193
```

## Labeling

You can create `FilePath label` format data by below command.

```
python label.py path/to/images/folder
```

