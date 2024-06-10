import glob
import os
import sys

def check_test_missing(model_path):
    test_path = glob.glob(os.path.join(model_path, "*", "*.json"))
    if len(test_path) == 0:
        return True
    else:
        return False

def check(base_dir):
    experiments = os.listdir(base_dir)
    for experiment_name in experiments:
        if experiment_name == "with_context_ablations":
            model_dirs = glob.glob(os.path.join(base_dir, experiment_name, "default", "*", "*", "last_checkpoint"))
        else:
            model_dirs = glob.glob(os.path.join(base_dir, experiment_name, "default", "*",  "last_checkpoint"))

        for epoch_path in model_dirs:
            model_dir = os.path.dirname(epoch_path)
            test_missing = check_test_missing(model_dir)
            if test_missing:
                print(model_dir)

if __name__ == "__main__":
    base_dir = sys.argv[1]
    check(base_dir)