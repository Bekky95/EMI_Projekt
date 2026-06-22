import os


def is_dataset_dir_existing(dataset_dir_name: str) -> bool:
    # search for project root
    path_to_root = "."
    while os.listdir(path_to_root).count("README.md") < 1:
        if len(path_to_root) == 1:
            path_to_root += "./"
        else:
            path_to_root += "../"

    # get project root
    root, _ = os.getcwd().split("src")
    # get complete path to dataset
    dataset_path = os.path.join("data", "dataset")
    searched_dir = root + dataset_path
    if os.listdir(searched_dir).count(dataset_dir_name) == 1:
        return True
    else:
        return False


if __name__ == "__main__":

    print(is_dataset_dir_existing("example"))
