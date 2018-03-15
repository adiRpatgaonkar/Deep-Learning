import sys, argparse
import train_net

def parse_arg():
    parser = argparse.ArgumentParser()

    # Config file
    parser.add_argument(
        "--CFG", 
        metavar='cfg_file', 
        type=str, 
        help="Config file for the model here."
    )
    # GPU
    parser.add_argument(
        "--GPU", 
        dest='GPU_ID', 
        type=int, 
        help="Specify GPU ID"
    )
    # Save the trained model
    parser.add_argument(
        "--SAVE",
        default='model.pkl',
        metavar='model file',
        type=str  ,
        help="Save the trained model (pickle file only)"
    )
    # Load pre-trained model
    parser.add_argument(
        "--LOAD",
        default='model.pkl',
        metavar='model file', 
        type=str, 
        help="Load a pre-trained model"
    )
    # Create a new model
    parser.add_argument(
        "--NEW", 
        action="store_true", 
        help="Create a new model"
    )
    # Fitting
    parser.add_argument(
        "--FIT", 
        action="store_true", 
        help="Model fits?"
    )
    # Training
    parser.add_argument(
        "--TRAIN", 
        action="store_true", 
        help="Train the model"
    )
    # Testing
    parser.add_argument(
        "--TEST", 
        action="store_true", 
        help="Test the model"
    )
    # Inference
    parser.add_argument(
        "--INFER", 
        action="store_true", 
        help="View inferences"
    )
    if len(sys.argv) is 1:
        parser.print_help()
        sys.exit(1)	

    global args
    global use_gpu
    args = parser.parse_args()

    if args.GPU_ID is None:
        use_gpu = False
    else:
        use_gpu = True

