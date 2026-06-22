import os


def is_dataset_dir_existing(dataset_dir_name: str) -> bool:
    # get project root
    root, _ = os.getcwd().split("src")
    # get complete path to dataset
    dataset_path = os.path.join("data", "dataset")
    searched_dir = root + dataset_path
    if os.listdir(searched_dir).count(dataset_dir_name) == 1:
        return True
    else:
        return False
