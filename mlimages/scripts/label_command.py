import os
import argparse
from mlimages.label import LabelingMachine

def make_parser():
    parser = argparse.ArgumentParser(description="Label the images")

    parser.add_argument("path", type=str, help="path to images data folder")
    parser.add_argument("--out", type=str, help="file path & name of the labeled images", default="")
    parser.add_argument("--label", type=int, help="label of the images", default=-1)
    parser.add_argument("--add", action="store_true", help="add to file")
    return parser


def main(args):
    machine = LabelingMachine(args.path)
    mode = "a" if not args.add else "w"
    if args.label < 0:
        machine.label_dir_auto(label_file=args.out, mode=mode)
    else:
        machine.label_dir(label=args.label, label_file=args.out, mode=mode)


if __name__ == "__main__":
    ps = make_parser()
    args = ps.parse_args()
    main(args)
