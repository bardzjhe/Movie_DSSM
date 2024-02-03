import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import pickle
from deepfm_model import *
from train import *
import pandas as pd
from src.config.paths import *
from src.config.constants import *

def get_feature_def(feat_meta):
    sparse_id_dims = []
    sparse_side_dims = []
    user_sparse, item_sparse, user_dense, item_dense = feat_meta["sparse"]["user"], feat_meta["sparse"]["item"], \
                                                       feat_meta["dense"]["user"], feat_meta["dense"]["item"]
    # parse id features
    sparse_id_dims.append(user_sparse["userid"][1])
    sparse_id_dims.append(item_sparse["movieid"][1])
    sparse_id_feat = ["userid", "movieid"]
    del user_sparse["userid"], item_sparse["movieid"]

    # parse other sparse features
    for col in user_sparse:
        sparse_side_dims.append(user_sparse[col][1])
    for col in item_sparse:
        sparse_side_dims.append(item_sparse[col][1])
    sparse_side_feat = list(user_sparse.keys()) + list(item_sparse.keys())

    # parse dense features
    dense_dim = len(user_dense) + len(item_dense)
    dense_feat = list(user_dense.keys()) + list(item_dense.keys())
    # print(f"sparse_id_dims: {sparse_id_dims}, sparse_side_dims: {sparse_side_dims}, dense_dim: {dense_dim}, sparse_id_feat: {sparse_id_feat}, sparse_side_feat: {sparse_side_feat}, dense_feat: {dense_feat}")
    return sparse_id_dims, sparse_side_dims, dense_dim, sparse_id_feat, sparse_side_feat, dense_feat

def load_data():
    # load feature metadata from a pickled file
    feature_meta = pickle.load(open(FEAT_META_PATH, "rb"))
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    return feature_meta, train_data, test_data

# Define a function to create PyTorch dataloaders for training and test sets
def get_dataloader(train_data, test_data, sparse_id_feat, sparse_side_feat, dense_feat):
    # Select the relevant features and label column from the training data
    train_data = train_data[sparse_id_feat + sparse_side_feat + dense_feat + ["label"]]
    # Select the relevant features and label column from the test data
    test_data = test_data[sparse_id_feat + sparse_side_feat + dense_feat + ["label"]]

    # Log the features being used
    print(f"load feat: {sparse_id_feat + sparse_side_feat + dense_feat}")

    # Create dataset objects for both training and testing
    train_dataset = MLDataSet(train_data, sparse_id_feat, sparse_side_feat, dense_feat, "label")
    test_dataset = MLDataSet(test_data, sparse_id_feat, sparse_side_feat, dense_feat, "label")

    # Create dataloaders to iterate over datasets in batches
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1)

    # Output the number of batches in the train and test dataloaders
    print(len(train_dataloader), len(test_dataloader))
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    # param
    device = args.device
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    epoch = args.epoch
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    # data
    feat_meta, train_data, test_data = load_data()
    sparse_id_dims, sparse_side_dims, dense_dim, sparse_id_feat, sparse_side_feat, dense_feat = get_feature_def(feat_meta)
    train_dataloader, test_dataloader = get_dataloader(train_data, test_data, sparse_id_feat, sparse_side_feat, dense_feat)

    # model
    model = DeepFactorizationMachineModel(sparse_id_dims, sparse_side_dims, dense_dim, embed_dim=16, embed_dim_side=2, mlp_dims=(4,), dropout=0.2).to(device)
    print(model)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=5, save_path=RANK_MODEL_PTH_PATH)

    # train
    for i in range(epoch):
        torch_train(model, optimizer, train_dataloader, criterion, device)
        auc = torch_test(model, test_dataloader, device)
        print(f"epoch {i}: test auc: {auc}")
        if not early_stopper.is_continuable(model, auc):
            print(f"test: best auc: {early_stopper.best_criterion}")
            break