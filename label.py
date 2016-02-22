import os
import argparse
from mlimages.model import API


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label the images")

    parser.add_argument("path", type=str, help="path to images data folder")
    parser.add_argument("--out", type=str, help="file path & name of the labeled images", default="")
    parser.add_argument("--label", type=int, help="label of the images", default=-1)
    parser.add_argument("--add", action="store_true", help="add to file")

    args = parser.parse_args()

    api = API(args.path)
    mode = "a" if not args.add else "w"
    if args.label < 0:
        api.label_dir_auto(label_path=args.out, mode=mode)
    else:
        api.label_dir(label=args.label, label_path=args.out, mode=mode)
