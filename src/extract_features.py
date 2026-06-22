# import mlcroissant as mlc
# import tensorflow_datasets as tfds
# import json
#
# # class ExtractFeatures:
# #     def __init__(self):
# #         pass
# #
# #     def extract_text_embeddings(self):
# #         pass
# #
# #     def extract_image_embeddings(self):
# #         pass
# #
# #     def save_embeddings(self):
# #         pass
#
# if __name__ == "__main__":
#
#     memetion_url = "https://huggingface.co/api/datasets/AshuReddy/memetion_dataset_7k/croissant"
#
#     # with open("../data/memetion_metadata.json", "w", encoding="utf-8") as f:
#     #     json.dump(ds.metadata.to_json(), f, indent=2, ensure_ascii=False)
#     #
#     # print("Metadaten gespeichert")
#     #
#     # https://github.com/mlcommons/croissant/tree/main
#
#     ds = mlc.Dataset(memetion_url)
#     json_metadata = ds.metadata.to_json()
#     print(json_metadata['name'])
#
#     # https://www.tensorflow.org/datasets/format_specific_dataset_builders#croissantbuilder
#     builder = tfds.dataset_builders.CroissantBuilder(
#         jsonld=memetion_url,
#         #record_set_ids=["fashion_mnist"],
#         file_format='array_record',
#     )
#     # builder = tfds.dataset_builders.CroissantBuilder(
#     #     # jsonld="https://raw.githubusercontent.com/mlcommons/croissant/main/datasets/0.8/huggingface-mnist/metadata.json",
#     #     jsonld=memetion_url,
#     #     file_format='array_record',
#     # )
#
#     builder.download_and_prepare(download_dir="../data/dataset")
#     datas = builder.as_data_source()

from datasets import load_dataset

dataset = load_dataset("AshuReddy/memetion_dataset_7k")
dataset.save_to_disk("../data/dataset")

