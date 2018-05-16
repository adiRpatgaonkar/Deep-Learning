import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "--bm",
    action="store_true",
    help="Save the best model")
parser.add_argument(
    "--fm",
    action="store_true",
    help="Save the final model")
parser.add_argument(
    "--gpu",
    dest="gpu_id",
    type=int,
    help="Specify GPU ID")
parser.add_argument(
    "--load",
    metavar="model_file",
    type=str,
    help="Load a model checkpoint")
parser.add_argument(
    "--train",
    action="store_true",
    help="Train the model")
parser.add_argument(
    "--test",
    action="store_true",
    help="Test the model")
parser.add_argument(
    "--infer",
    action="store_true",
    help="Inference the model")
parser.add_argument(
    "--model",
    dest="mid",
    type=str,
    help="Model ID used to build")
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="Maximum number of epochs to run")
parser.add_argument(
    "--lr",
    type=float,
    default=0,
    help="Base learning rate")
parser.add_argument(
    "--bs",
    type=int,
    default=1,
    help="Batch size for training/testing")

args = parser.parse_args()
if type(args.gpu_id) == int:
    args.gpu_id = str(args.gpu_id)

print("\n" + str(args) + "\n")

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit("Nothing to do")
