from Dataset.segmentation_OSF.Dataset import OSF_data_loaders
from Dataset.modelnet.Dataloader import modelnet_data_loaders

# Factory to choose the dataset
DATASET_FACTORY = {
    'modelnet': modelnet_data_loaders,
    'OSF': OSF_data_loaders,
}

def get_dataset_loader(dataset_name, args):
    if dataset_name in DATASET_FACTORY:
        return DATASET_FACTORY[dataset_name](args)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")