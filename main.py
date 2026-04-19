from configs import Config
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--max_length", type=str)

    return parser.parse_args()

args = parse_arguments()


if __name__ == "__main__":
    print(f"Running model : {args.model}")

    #Load model 

    #Start Fastapi server

    