
import torch
import torch.nn as nn
from models.transformer import Transformer  
import torchvision.models as models

class MLP(nn.Module):
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
    def __init__(self, num_classes, num_queries=100, hidden_dim=256, nheads=8,
                 dim_feedforward=512, enc_layers=3, dec_layers=3, dropout=0.3, pretrained_weights_path=None):
        super(DETR, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Backbone: ResNet50
        # backbone = resnet50(pretrained=True)
        backbone = models.resnet50(pretrained=False)
        if pretrained_weights_path:
            state_dict = torch.load(pretrained_weights_path)
            backbone.load_state_dict(state_dict)
        else:
            backbone = models.resnet50(pretrained=True)
    
        backbone_layers = list(backbone.children())[:-2]  # Remove avgpool and fc
        self.backbone = nn.Sequential(*backbone_layers)
        backbone_output_dim = 2048  # ResNet50's final layer output channels

        # Project backbone features to hidden_dim
        self.input_proj = nn.Conv2d(backbone_output_dim, hidden_dim, kernel_size=1)

        # Transformer
        self.transformer = Transformer(embed_dim=hidden_dim,
                                           num_heads=nheads,
                                           ff_dim=dim_feedforward,
                                           num_encoder_layers=enc_layers,
                                           num_decoder_layers=dec_layers,
                                           dropout=dropout,
                                           max_len=5000)

        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Output prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


    def forward(self, x):
        """
        x: Input images tensor of shape (batch_size, 3, H, W)
        """
        # Backbone feature extraction
        features = self.backbone(x)  # (batch_size, backbone_output_dim, H', W')

        # Project backbone features to hidden_dim
        src = self.input_proj(features)  # (batch_size, hidden_dim, H', W')
        batch_size, hidden_dim, H, W = src.shape

        # Flatten spatial dimensions
        src = src.flatten(2).permute(0, 2, 1)  # (batch_size, H'*W', hidden_dim)

        # Create query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, num_queries, hidden_dim]
        tgt = torch.zeros_like(query_embed)  # [batch_size, num_queries, hidden_dim]

        # Pass through Transformer
        hs = self.transformer(src, tgt)  # [batch_size, num_queries, hidden_dim]

        # Output heads
        outputs_class = self.class_embed(hs)  # [batch_size, num_queries, num_classes + 1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [batch_size, num_queries, 4]

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}