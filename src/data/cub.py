import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


class CUB200(Dataset):

    base_folder: str = 'CUB_200_2011/images'
    url: str = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    filename: str = 'CUB_200_2011.tgz'
    md5: str = "97eceeb196236b17998738112f37df78"
    non_concept_columns: list[str] = ["class_name", "class_id", "is_training_image", "image_name", "image_id"]

    def __init__(self, root, split: str = "train", transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.split = split

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')


    def _load_metadata(self):
        if not os.path.isfile(os.path.join(self.root, "CUB_200_2011", "metadata.csv")):
            print("Preparing metadata file")
            prepare_metadata(self.root)
        
        fp = os.path.join(self.root, "CUB_200_2011", "metadata.csv")
        self.data = pd.read_csv(fp)

        if self.split == "train":
            self.data = self.data[self.data.is_training_image == 1]
        elif self.split == "test":
            self.data = self.data[self.data.is_training_image == 0]

        self.concept_columns = [
            col for col in self.data.columns.tolist() 
            if col not in self.non_concept_columns
        ]

        self.idx_to_class = {
            idx: class_name for idx, class_name in enumerate(self.data.class_name.unique())
        }
        self.idx_to_concept = {
            idx: concept_name for idx, concept_name in enumerate(self.concept_columns)
        }
            

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception as e:
            print("error: ", e)
            return False
        for _, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.image_name)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.image_name)
        label = sample.class_id - 1  # class ids start at 1, so shift to 0
        concepts = sample[self.concept_columns].values.astype(float)

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, concepts
    

def prepare_metadata(data_root):
    import pandas as pd

    attributes = f"{data_root}/CUB_200_2011/attributes/attributes.txt"
    classes = f"{data_root}/CUB_200_2011/classes.txt"
    class_labels = f"{data_root}/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"
    image_attributes = f"{data_root}/CUB_200_2011/attributes/image_attribute_labels.txt"
    image_class_labels = f"{data_root}/CUB_200_2011/image_class_labels.txt"
    images = f"{data_root}/CUB_200_2011/images.txt"
    split = f"{data_root}/CUB_200_2011/train_test_split.txt"

    images_df = pd.read_csv(images, sep=" ", header=None, names=["image_id", "image_name"])
    labels_df = pd.read_csv(image_class_labels, sep=" ", header=None, names=["image_id", "class_id"])
    split_df = pd.read_csv(split, sep=" ", header=None, names=["image_id", "is_training_image"])
    attributes_df = pd.read_csv(attributes, sep=" ", header=None, names=["attribute_id", "attribute"])
    classes_df = pd.read_csv(classes, sep=" ", header=None, names=["class_id", "class_name"])
    class_attributes_df = pd.read_csv(class_labels, sep=" ", header=None)

    data = []
    with open(image_attributes, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
                
            parts = line.split(maxsplit=4)
            
            if len(parts) == 5:
                data.append(parts)
            else:
                print(f"Skipping: {line}")

    try:
        image_attributes_df = pd.DataFrame(data, columns=['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time'])
        image_attributes_df.drop(["time"], inplace=True, axis=1)
    except Exception as e:
        print(f"\nAn error occurred when creating the DataFrame: {e}")
        return None

    image_label = images_df.merge(labels_df, on="image_id")
    image_label_split = image_label.merge(split_df, on="image_id")

    image_attributes_df['attribute_id'] = image_attributes_df['attribute_id'].astype(str)
    attributes_df['attribute_id'] = attributes_df['attribute_id'].astype(str)
    merged_images_df = pd.merge(
        image_attributes_df, 
        attributes_df, 
        on='attribute_id'
    )
    merged_images_df['is_present'] = pd.to_numeric(merged_images_df['is_present'])
    merged_images_df['image_id'] = pd.to_numeric(merged_images_df['image_id'])

    image_features_df = pd.pivot_table(
        merged_images_df,
        index='image_id',
        columns='attribute',
        values='is_present',
        fill_value=0,
    )

    image_features_df = image_features_df.reset_index().rename_axis(None, axis=1)

    almost_full = image_features_df.merge(image_label_split, on="image_id")
    full = almost_full.merge(classes_df, on="class_id")
    cols = sorted(full.columns)
    full = full[cols]
    full.to_csv(f"{data_root}/CUB_200_2011/metadata.csv", index=False)

    class_attribute_df = pd.concat([classes_df, class_attributes_df], axis=1)
    class_attribute_df.to_csv(f"{data_root}/CUB_200_2011/class_attribute_mapping.csv", index=False)