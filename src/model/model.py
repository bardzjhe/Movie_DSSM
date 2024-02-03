
import torch
import numpy as np

# 更多model可以在这里找到

# Define a Linear part for the Factorization Machine (FM) model
class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        # Embedding layer for the linear part with one weight per feature
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        # Bias term for the linear part
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        # Calculate offsets for each field to map input fields to the correct embedding
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        Forward pass for the linear part of the FM model
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # Apply offsets to the input feature indices and pass them through the embedding layer
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # Sum up the embeddings and add the bias
        return torch.sum(self.fc(x), dim=1) + self.bias


# Define embeddings for the features used in the FM part
class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # Embedding layer for the FM part with an embedding vector per feature
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        # Initialize the embedding weights using Xavier uniform initialization
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        # Calculate offsets for each field to map input fields to the correct embedding
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        Forward pass for the embedding part of the FM model
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # Apply offsets to the input feature indices and pass them through the embedding layer
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # Return the embeddings
        return self.embedding(x)


# Define the Factorization Machine part for capturing feature interactions
class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        # Whether to sum the results after computing interactions
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        Forward pass for the FM part to capture interactions
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)`` or a list of such tensors
        """
        # Compute pairwise interactions using the formula
        if isinstance(x, list):
            ix = None
            for xx in x:
                # Compute the square of the sum and sum of squares
                square_of_sum = torch.sum(xx, dim=1) ** 2
                sum_of_square = torch.sum(xx ** 2, dim=1)
                # Compute the interaction term for the current feature set
                if ix is None:
                    ix = square_of_sum - sum_of_square
                else:
                    # Concatenate interaction terms for different feature sets
                    ix = torch.cat([ix, square_of_sum - sum_of_square], dim=1)
        else:
            # Compute the interaction term for a single feature set
            square_of_sum = torch.sum(x, dim=1) ** 2
            sum_of_square = torch.sum(x ** 2, dim=1)
            ix = square_of_sum - sum_of_square

        # If reduce_sum is true, sum over the feature dimension
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        # Return half of the interaction term (following the FM equation)
        return 0.5 * ix


# Define a Multi-Layer Perceptron for learning high-order feature interactions
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        # Construct the layers of the MLP
        for embed_dim in embed_dims:
            # Add a dense layer
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # Add batch normalization
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            # Add ReLU activation function
            layers.append(torch.nn.ReLU())
            # Add dropout for regularization
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        # If output_layer is True, add a final dense layer with a single output
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        # Combine all layers into```python
        # a sequential model
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for MLP
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        # Pass the input through the MLP to get the output
        return self.mlp(x)


# Define the Factorization Machine-based model
class FactorizationMachineModel(torch.nn.Module):
    """
    A PyTorch implementation of Factorization Machine.
    Reference: S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # Initialize embeddings for features
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # Initialize the linear part of the FM model
        self.linear = FeaturesLinear(field_dims)
        # Initialize the Factorization Machine part for capturing feature interactions
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        Forward pass for the FM model
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # Compute the linear and FM part then combine them
        x = self.linear(x) + self.fm(self.embedding(x))
        # Apply sigmoid to the combined output to get probabilities
        return torch.sigmoid(x.squeeze(1))


# Define the DeepFM model that combines FM and deep learning
class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A PyTorch implementation of DeepFM.
    Reference: H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, sparse_id_dims, sparse_side_dims, dense_dim, embed_dim, embed_dim_side, mlp_dims, dropout):
        super().__init__()
        # Initialize the linear part of the model
        self.linear = FeaturesLinear(sparse_id_dims + sparse_side_dims)
        # Initialize embeddings for ID features
        self.embedding = FeaturesEmbedding(sparse_id_dims, embed_dim)
        # Calculate the output dimension of ID embeddings
        self.embed_output_dim = len(sparse_id_dims) * embed_dim
        # Count the number of ID fields and side information fields
        self.num_fields = len(sparse_id_dims)
        self.num_fields_side = len(sparse_side_dims)
        # Set the dimension for dense features
        self.dense_dim = dense_dim

        # If there are side information fields, initialize embeddings for them
        if self.num_fields_side > 0:
            self.embedding_side = FeaturesEmbedding(sparse_side_dims, embed_dim_side)
            self.embed_output_dim_side = len(sparse_side_dims) * embed_dim_side

        # If there are dense features, initialize an MLP for them
        if self.dense_dim > 0:
            self.mlp_dense = MultiLayerPerceptron(dense_dim, mlp_dims, dropout)
        # Initialize the Factorization Machine part for capturing feature interactions
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, xx):
        """
        Forward pass for the DeepFM model
        :param xx: Long tensor of size ``(batch_size, num_sparse_id_fields + num_sparse_side_fields + dense_dim)``
        """
        # Split the input into sparse ID, sparse side information, and dense features
        x_sparse_id, x_sparse_side, x_dense = xx[:, :self.num_fields], \
            xx[:, self.num_fields:self.num_fields + self.num_fields_side], \
            xx[:, self.num_fields + self.num_fields_side:]
        # Convert the sparse ID features to integers and get embeddings
        x_sparse_id = x_sparse_id.to(torch.int32)
        embed_x_id = self.embedding(x_sparse_id)

        # If there are side information fields, convert them to integers and get embeddings
        if self.num_fields_side > 0:
            x_sparse_side = x_sparse_side.to(torch.int32)
            embed_x_side = self.embedding_side(x_sparse_side)
            # Concatenate the sparse ID and side information embeddings
            x_sparse = torch.cat([x_sparse_id, x_sparse_side], dim=1)
            embed_x = [embed_x_id, embed_x_side]
        else:
            x_sparse = x_sparse_id
            embed_x = embed_x_id

        # Convert dense features to float and compute the linear, FM, and MLP parts
        x_dense = x_dense.to(torch.float32)
        y = self.linear(x_sparse) + self.fm(embed_x)
        if self.dense_dim > 0:
            y += self.mlp_dense(x_dense)
        # Apply sigmoid to the output and squeeze to remove extra dimensions
        return torch.sigmoid(y.squeeze(1))