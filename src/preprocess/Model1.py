import torch
import numpy as np

# Define linear feature transformation module
class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        # Embedding layer for linear feature transformation
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        # Bias term for linear transformation
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        # Compute offsets to re-index the fields
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        Forward method for linear features.
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # Adjust indices by offsets and apply embedding
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # Sum the embeddings and add bias
        return torch.sum(self.fc(x), dim=1) + self.bias

# Define embedding feature transformation module
class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # Embedding layer for nonlinear feature transformation
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        # Compute offsets to re-index the fields
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        # Initialize the embedding weights
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        Forward method for embedded features.
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # Adjust indices by offsets and apply embedding
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

# Define Factorization Machine module
class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        # Option to reduce the sum of the output
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        Forward method for Factorization Machines.
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)`` or a list of such tensors
        """
        # Calculate interactions differently for list of tensors or a single tensor
        if isinstance(x, list):
            ix = None
            # Calculate interactions for each tensor in the list
            for xx in x:
                square_of_sum = torch.sum(xx, dim=1) ** 2
                sum_of_square = torch.sum(xx ** 2, dim=1)
                if ix is None:
                    ix = square_of_sum - sum_of_square
                else:
                    ix = torch.cat([ix, square_of_sum - sum_of_square], dim=1)
        else:
            # Calculate interactions for a single tensor
            square_of_sum = torch.sum(x, dim=1) ** 2
            sum_of_square = torch.sum(x ** 2, dim=1)
            ix = square_of_sum - sum_of_square
        # Reduce sum if required
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        # Factorization Machine formula
        return 0.5 * ix

# Define Multi-Layer Perceptron module
class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        # Create MLP layers with batch normalization, ReLU activation, and dropout
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        # Optionally add an output layer
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        # Sequentialize the defined layers
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward method for the MLP.
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

# Define Factorization Machine Model
class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # Initialize embedding, linear, and FM components
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        Forward method for the Factorization Machine Model.
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # Get the output from the linear part and the FM part
        x = self.linear(x) + self.fm(self.embedding(x))
        # Apply sigmoid to the output to get probabilities
        return torch.sigmoid(x.squeeze(1))

# Define Deep Factorization Machine Model
class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, sparse_id_dims, sparse_side_dims, dense_dim, embed_dim, embed_dim_side, mlp_dims, dropout):
        super().__init__()
        # Initialize linear part for sparse and side information
        self.linear = FeaturesLinear(sparse_id_dims + sparse_side_dims)
        # Initialize embedding for the sparse ID fields
        self.embedding = FeaturesEmbedding(sparse_id_dims, embed_dim)
        # Calculate the output dimension for embeddings
        self.embed_output_dim = len(sparse_id_dims) * embed_dim
        self.num_fields = len(sparse_id_dims)
        self.num_fields_side = len(sparse_side_dims)
        self.dense_dim = dense_dim

        # If side information is present, initialize side embeddings
        if self.num_fields_side > 0:
            self.embedding_side = FeaturesEmbedding(sparse_side_dims, embed_dim_side)
            self.embed_output_dim_side = len(sparse_side_dims) * embed_dim_side

        # If dense features are present, initialize MLP for dense features
        if self.dense_dim > 0:
            self.mlp_dense = MultiLayerPerceptron(dense_dim, mlp_dims, dropout)
        # Initialize FM part
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, xx):
        """
        Forward method for the Deep Factorization Machine Model.
        :param xx: Long tensor of size ``(batch_size, num_sparse_id_fields + num_sparse_side_fields + dense_dim)``
        """
        # Split the input tensor into sparse ID, side, and dense parts
        x_sparse_id, x_sparse_side, x_dense = xx[:, :self.num_fields], \
                                               xx[:, self.num_fields:self.num_fields+self.num_fields_side], \
                                               xx[:, self.num_fields+self.num_fields_side:]
        # Convert sparse ID and side fields to int32 for embedding lookup
        x_sparse_id = x_sparse_id.to(torch.int32)
        embed_x_id = self.embedding(x_sparse_id)

        if self.num_fields_side > 0:
            x_sparse_side = x_sparse_side.to(torch.int32)
            embed_x_side = self.embedding_side(x_sparse_side)
            x_sparse = torch.cat([x_sparse_id, x_sparse_side], dim=1)
            embed_x = [embed_x_id, embed_x_side]
        else:
            x_sparse = x_sparse_id
            embed_x = embed_x_id

        # Convert dense features to float32 for subsequent operations
        x_dense = x_dense.to(torch.float32)
        # Combine the outputs from linear, FM, and (if present) MLP parts
        y = self.linear(x_sparse) + self.fm(embed_x)
        if self.dense_dim > 0:
            y += self.mlp_dense(x_dense)
        # Apply sigmoid to the output to get probabilities
        return torch.sigmoid(y.squeeze(1))