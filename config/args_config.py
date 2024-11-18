import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")
    # Training args
    ## loops
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")

    ## sampling
    parser.add_argument('--sampling', type=str, default='fps', help="Sampling technique")
    parser.add_argument('--n_centroids', type=int, default=2048, help="centroids")
    parser.add_argument('--nsamples', type=int, default=16, help="sample points")
    parser.add_argument('--radius', type=float, default=0.1, help="radius of ball query")
    parser.add_argument('--knn', type=int, default=16, help="neighbours for Dyanmic Graph contruction")

    ## Model and Tasks
    parser.add_argument('--k', type=int, default=33, help="Number classes")
    parser.add_argument('--model', type=str, default="PointNet", help="Select the model")
    parser.add_argument('--mode', type=str, default="segmentation", help="Problems ex:- segmentaion, classification")
    parser.add_argument('--loss', type=str, default="crossentropy", help="Select the suitable loss function ex:- focal, dice, crossentropy, chamferdistance, hausdorff")
    parser.add_argument('--pretrained', type=str, default=None, help="Path to pretrained models")

    ## Training hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")
    parser.add_argument('--gamma', type=float, default=0.5, help="Learning rate decay")
    
    parser.add_argument('--rotat', type=float, default=0.25, help="rotation invariates")
    parser.add_argument('--trans', type=float, default=0.5, help="translation invariantes")

    # Datasets args
    ## Pathes
    parser.add_argument('--path', type=str, default="dataset", help="Path of the dataset")
    parser.add_argument('--Dataset', type=str, default="OSF", help="Which Dataset?")
    parser.add_argument('--output', type=str, default="output", help="Output path")
    parser.add_argument('--test_ids', type=str, default="private-testing-set.txt", help="Path of the ids dataset for testing")
    parser.add_argument('--p', type=int, default=6, help="data parts")

    ## DataLoader    
    parser.add_argument('--num_workers', type=int, default=4, help="Number of Workers")

    ## Processing on the data
    parser.add_argument('--clean', action='store_true', help="Clean some of the gingave points")
    parser.add_argument('--rigid_augmentation_train', action='store_true', help="More Transformations")
    parser.add_argument('--rigid_augmentation_test', action='store_true', help="More Transformations")

    return parser.parse_args()
