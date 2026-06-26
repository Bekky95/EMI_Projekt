import matplotlib.pyplot as plt
import random


def plot_14_random_entries_from_dataset(cur_datensatz):
    fig, axes = plt.subplots(2, 7, figsize=(14, 5))
    axes = axes.flatten()

    indices = random.sample(range(len(cur_datensatz['train'])), 14)

    for ax, idx in zip(axes, indices):
        img = cur_datensatz['train'][idx]['images'][0]
        label = cur_datensatz['train'][idx]['messages'][0]['content']

        ax.imshow(img, cmap='gray')
        ax.set_title(label)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
