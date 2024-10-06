import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")

    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of Workers")

    parser.add_argument('--sampling', type=str, default='fps', help="Sampling technique")

    parser.add_argument('--n_centroids', type=int, default=2048, help="centroids")
    parser.add_argument('--nsamples', type=int, default=16, help="sample points")
    parser.add_argument('--radius', type=float, default=0.1, help="radius of ball query")
    parser.add_argument('--knn', type=int, default=16, help="neighbours for Dyanmic Graph contruction")

    parser.add_argument('--path', type=str, default="dataset", help="Path of the dataset")
    parser.add_argument('--Dataset', type=str, default="OSF", help="Which Dataset?")

    parser.add_argument('--output', type=str, default="output", help="Output path")

    parser.add_argument('--test_ids', type=str, default="private-testing-set.txt", help="Path of the ids dataset for testing")
    parser.add_argument('--p', type=int, default=3, help="data parts")

    parser.add_argument('--k', type=int, default=33, help="Number classes")
    parser.add_argument('--model', type=str, default="PointNet", help="Select the model")
    parser.add_argument('--mode', type=str, default="segmentation", help="Problems ex:- segmentaion, classification")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")

    return parser.parse_args()
