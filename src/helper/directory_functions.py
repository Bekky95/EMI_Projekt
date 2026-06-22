import os


def is_dataset_dir_existing(dataset_dir_name: str) -> bool:
    """
    checks if dataset directory (dataset_dir_name) exists in "./data/dataset"
    """
    root = get_root()

    # get complete path to dataset
    dataset_path = os.path.join("data", "dataset")
    searched_dir = os.path.join(root, dataset_path)
    if os.listdir(searched_dir).count(dataset_dir_name) == 1:
        return True
    else:
        return False


def get_root() -> str:
    # get project root
    cwd = os.getcwd()
    cwd_len = len(cwd.split("src"))
    root = ""

    if cwd_len < 1:
        root, _ = os.getcwd().split("src")
    elif cwd_len == 1:
        root = cwd

    return root


def create_dir_name(dataset_name):
    """
    Takes name of dataset, turns it into snake_case directory name

    e.g.:
    "Leonardo6/memotion" -> Leonardo6_memotion_
    """
    name_list = dataset_name.split("/")
    dir_name = ""
    for items in name_list:
        dir_name += (items + "_")

    return dir_name
