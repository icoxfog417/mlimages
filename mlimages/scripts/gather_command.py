import os
import argparse

def make_parser():
    parser = argparse.ArgumentParser(description="Gather Image Dataset")

    parser.add_argument("-path", type=str, help="path to data folder", default="")
    parser.add_argument("-parallel", type=int, help="number of process to download", default=-1)
    parser.add_argument("-limit", type=int, help="upper limit of image count", default=-1)

    parser.add_argument("-imagenet", action="store_true", help="gather imagenet dataset")
    parser.add_argument("--wnid", type=str, help="target wnid")
    parser.add_argument("--subset", action="store_true", help="download child subset")

    return parser

def main(args):
    current = os.getcwd()
    path = args.path if args.path else os.path.join(current, "data/")
    parallel = args.parallel
    limit = args.limit

    api = None
    if args.imagenet:
        from mlimages.gather.imagenet import ImagenetAPI
        api = ImagenetAPI(path, parallel=parallel, limit=limit)
        api.logger.info("start to gather the ImageNet images.")
        api.gather(wnid=args.wnid, include_subset=args.subset)

if __name__ == "__main__":
    ps = make_parser()
    args = ps.parse_args()
    main(args)
