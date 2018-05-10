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

args = parser.parse_args()
if args.gpu_id:
    args.gpu_id = str(args.gpu_id)

print("\n" + str(args) + "\n")

if len(sys.argv) == 1:
    sys.exit("Nothing to do")
