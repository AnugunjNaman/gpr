import torch
import torch.nn as nn
from models.transformer import Transformer
import torchvision.models as models

class MLP(nn.Module):
    """Multilayer Perceptron (MLP) for bounding box regression."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DETR(nn.Module):
    """DEtection TRansformer (DETR) model for object detection."""

    def __init__(self, num_classes, num_queries=100, hidden_dim=256, nheads=8,
                 dim_feedforward=512, enc_layers=3, dec_layers=3, dropout=0.3, pretrained_weights_path=None):
        """
        Args:
            num_classes (int): Number of object classes (excluding background).
            num_queries (int): Number of object queries in the transformer decoder.
            hidden_dim (int): Dimensionality of the transformer embeddings.
            nheads (int): Number of attention heads in the transformer.
            dim_feedforward (int): Feedforward layer dimensionality in transformer.
            enc_layers (int): Number of transformer encoder layers.
            dec_layers (int): Number of transformer decoder layers.
            dropout (float): Dropout rate.
            pretrained_weights_path (str, optional): Path to pretrained backbone weights.
        """
        super(DETR, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Initialize the backbone (ResNet-50)
        self.backbone = self._build_backbone(pretrained_weights_path)

        # Projection layer to match Transformer hidden dimension
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # Transformer
        self.transformer = Transformer(
            embed_dim=hidden_dim,
            num_heads=nheads,
            ff_dim=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dropout=dropout,
            max_len=5000
        )

        # Object query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Output heads: Classification and Bounding Box Prediction
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def _build_backbone(self, pretrained_weights_path):
        """Initializes the ResNet-50 backbone with optional pretrained weights."""
        backbone = models.resnet50(weights=None)

        if pretrained_weights_path:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove fully connected layers and pooling
        return nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        """
        Forward pass for DETR.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, 3, H, W).

        Returns:
            dict: A dictionary with:
                - 'pred_logits': Class logits, shape (batch_size, num_queries, num_classes + 1).
                - 'pred_boxes': Normalized bounding boxes, shape (batch_size, num_queries, 4).
        """
        # Feature extraction via backbone
        features = self.backbone(x)  # (batch_size, 2048, H', W')

        # Project features to transformer-compatible hidden_dim
        src = self.input_proj(features)  # (batch_size, hidden_dim, H', W')
        batch_size, _, H, W = src.shape

        # Flatten spatial dimensions for Transformer input
        src = src.flatten(2).permute(0, 2, 1)  # (batch_size, H'*W', hidden_dim)

        # Create learnable query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, num_queries, hidden_dim)
        tgt = torch.zeros_like(query_embed)  # Initial decoder input

        # Transformer forward pass
        hs = self.transformer(src, tgt)  # (batch_size, num_queries, hidden_dim)

        # Output predictions
        outputs_class = self.class_embed(hs)  # (batch_size, num_queries, num_classes + 1)
        outputs_coord = self.bbox_embed(hs).sigmoid()  # (batch_size, num_queries, 4)

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
