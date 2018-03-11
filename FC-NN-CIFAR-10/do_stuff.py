import argparse
import train_net

def parse_arg():

    print("Parsing arguments")
    parser = argparse.ArgumentParser()

    # Config file
    parser.add_argument("CFG", type=str, help="Config file for the model here.")
    # GPU
    parser.add_argument("--GPU", metavar='GPU_ID', type=int, help="Specify GPU ID")
    # Fitting
    parser.add_argument("--FIT", action="store_true", help="Model fits?")
    # Training
    parser.add_argument("--TRAIN", action="store_true", help="Train the model")
    # Testing
    parser.add_argument("--TEST", action="store_true", help="Test the model")
    # Inference
    parser.add_argument("--INFER", action="store_true", help="View inferences")
    # Load pre-trained model
    parser.add_argument("--LOAD", metavar='IMPORT_MODEL', type=str, help="Load a pre-trained model")
    # Save the trained model
    parser.add_argument("--SAVE", type=str, help="Save the trained model")

    global args
    args = parser.parse_args()



