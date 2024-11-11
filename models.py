# %% [code]
"""
Model Architectures for NFL Big Data Bowl 2025 prediction task

This module defines neural network architectures for predicting ouptuts in NFL Plays. It includes a generalized transformer-based model for
processing sports tracking data and a LightningModule wrapper for training and evaluation.

Classes:
    SportsTransformer: Generalized Transformer-based model for sports tracking data.
    SportsTransformerLitModel: LightningModule wrapper for shared training functionality.
"""

from typing import Any
import torch
from torch import Tensor, nn, squeeze
from torch.optim import AdamW
from pytorch_lightning import LightningModule

torch.set_float32_matmul_precision("medium")

class SportsTransformer(nn.Module):
    """
    Transformer model architecture for processing sports tracking data.

    This model leverages self-attention mechanisms to process player tracking data, capturing
    the spatial and temporal relationships between players on the field.

    Attributes:
        feature_len (int): Number of input features per player.
        model_dim (int): Dimension of the model's internal representations.
        num_layers (int): Number of transformer encoder layers.
        dropout (float): Dropout rate for regularization.
        hyperparams (dict): Dictionary storing hyperparameters of the model.
    """

    def __init__(
        self,
        feature_len: int,
        model_dim: int = 32,
        num_layers: int = 2,
        output_dim: int = 7,
        dropout: float = 0.3,
    ):
        """
        Initialize the SportsTransformer.

        Args:
            feature_len (int): Number of input features per player.
            model_dim (int): Dimension of the model's internal representations.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        dim_feedforward = model_dim * 4
        num_heads = min(16, max(2, 2 * round(model_dim / 64)))  # Attention is optimized for even number of heads

        self.hyperparams = {
            "model_dim": model_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dim_feedforward": dim_feedforward,
        }

        # Normalize input features
        self.feature_norm_layer = nn.BatchNorm1d(feature_len)

        # Embed input features to model dimension
        self.feature_embedding_layer = nn.Sequential(
            nn.Linear(feature_len, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        # Transformer Encoder
        # This component applies multiple layers of self-attention and feed-forward networks
        # to process player data in a permutation-equivariant manner.
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Pool across player dimension
        self.player_pooling_layer = nn.AdaptiveAvgPool1d(1)

        # Task-specific Decoder to predict output.
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(model_dim // 4),
            nn.Linear(model_dim // 4, output_dim),  # Adjusted to match target shape
        )
 
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SportsTransformer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_players, feature_len].

        Returns:
            Tensor: Predicted output of shape [batch_size, 22].
        """
        # x: [B: batch_size, P: # of players, F: feature_len]
        B, P, F = x.size()

        # Normalize features
        x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B,P,F] -> [B,P,F]

        # Embed features
        x = self.feature_embedding_layer(x)  # [B,P,F] -> [B,P,M: model_dim]

        # Apply transformer encoder
        x = self.transformer_encoder(x)  # [B,P,M] -> [B,P,M]

        # Pool over player dimension
        x = squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)  # [B,M,P] -> [B,M]

        # Decode to predict output
        x = self.decoder(x)  # [B,M] -> [B, output_dim]
        
        return x

class SportsTransformerLitModel(LightningModule):
    """
    Lightning module for training and evaluating models.

    This class wraps the SportsTransformer for training, validation, and testing
    using PyTorch Lightning, providing a structured training loop, optimization, and logging.

    Attributes:
        feature_len (int): Number of input features per player.
        batch_size (int): Batch size for training and evaluation.
        model (SportsTransformer): The transformer-based model.
        learning_rate (float): Learning rate for the optimizer.
        loss_fn (nn.Module): Loss function used for training.
    """

    def __init__(
        self,
        feature_len: int,
        batch_size: int,
        model_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize the SportsTransformerLitModel.

        Args:
            feature_len (int): Number of input features.
            batch_size (int): Batch size for training and evaluation.
            model_dim (int): Dimension of the model's internal representations.
            num_layers (int): Number of layers in the model.
            dropout (float): Dropout rate for regularization.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.feature_len = feature_len
        self.model = SportsTransformer(
            feature_len=self.feature_len, model_dim=model_dim, num_layers=num_layers, dropout=dropout, output_dim=output_dim
        )
        self.example_input_array = torch.randn((batch_size, 22, self.feature_len))

        self.learning_rate = learning_rate
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.hparams["params"] = self.num_params
        for k, v in self.model.hyperparams.items():
            self.hparams[k] = v

        self.save_hyperparameters()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        if isinstance(x, list):
            x = torch.stack(x)
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Perform a single training step.

        Args:
            batch (tuple[Tensor, Tensor]): Batch of input features and target locations.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed loss for the batch.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Validation step for the model.

        Args:
            batch (tuple[Tensor, Tensor]): Batch of input and target tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed loss.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """
        Prediction step for the model.

        Args:
            batch (tuple[Tensor, Tensor]): Batch of input and target tensors.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the dataloader.

        Returns:
            Tensor: Predicted output tensor.
        """
        x, _ = batch
        if isinstance(x, list):
            x = torch.stack(x)
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self) -> AdamW:
        """
        Configure the optimizer for training.

        Returns:
            AdamW: Configured optimizer.
        """
        return AdamW(self.parameters(), lr=self.learning_rate)
    