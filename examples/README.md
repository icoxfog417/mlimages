# Examples

`chainer_alex.py` is the script that demonstrate below process.

* gather images from ImageNet (I gathered [`domestic cat (wnid=n02121808)`](http://image-net.org/explore?wnid=n02121808) and its subsets).
 * 0 abyssinian
 * 1 alley_cat
 * 2 angora
 * 3 burmese_cat
 * 4 egyptian_cat
 * 5 kitty
 * 6 manx
 * 7 mouser
 * 8 persian_cat
 * 9 siamese_cat
 * 10 tabby
 * 11 tiger_cat
 * 12 tom
 * 13 tortoiseshell
 * 14 ROOT(domestic cat)
* label data according to folder structure
* see gathered images
* make mean image and train the model
* predict by trained model

You can execute it from console (please don't forget to activate your virtual environments that installed `requirements.txt`).

`python chainer_alex.py`

(please see usage by `-h`)

**You can use my trained model for your fine tuning (accuracy is about 80~90%).**  
It is stored in `examples/pretrained` by [`git lfs`](https://git-lfs.github.com/) (mean image and label definition is there too).
