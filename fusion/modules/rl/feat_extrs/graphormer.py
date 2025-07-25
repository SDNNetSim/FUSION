import torch
from torch_geometric.nn import TransformerConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# TODO: (version 5.5-6) Add params to optuna

class GraphTransformerExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Graph Transformer with SB3.
    """

    def __init__(self, obs_space, emb_dim, heads, layers):
        num_paths = obs_space["path_masks"].shape[0]
        in_dim = obs_space["x"].shape[1]
        out_per_head = emb_dim // heads
        conv_out_dim = heads * out_per_head
        super().__init__(obs_space, features_dim=emb_dim * num_paths)

        self.convs_matrix = torch.nn.ModuleList([
            TransformerConv(
                in_channels=(in_dim if i == 0 else conv_out_dim),
                out_channels=out_per_head,
                heads=heads,
                concat=True
            )
            for i in range(layers)
        ])
        self.readout_obj = torch.nn.Linear(conv_out_dim, emb_dim)

    def forward(self, obs):
        """
        Convert observation to feature vector.
        """
        x_list = obs["x"]  # [B, N, F] or [N, F]
        ei_list = obs["edge_index"].long()  # [B, 2, E] or [2, E]
        masks_list = obs["path_masks"]  # [B, k, E] or [k, E]

        # Handle batch dimension
        if x_list.dim() == 3:
            batch_size = x_list.size(0)
            if batch_size > 1:
                outputs = []
                for b in range(batch_size):
                    xb = x_list[b]
                    eib = ei_list[b] if ei_list.dim() == 3 else ei_list
                    mb = masks_list[b] if masks_list.dim() == 3 else masks_list
                    yb = xb
                    for conv in self.convs_matrix:
                        yb = conv(yb, eib).relu()
                    src_idx, dst_idx = eib
                    edge_emb_b = (yb[src_idx] + yb[dst_idx]) * 0.5
                    path_emb_b = mb @ edge_emb_b
                    pv_b = self.readout_obj(path_emb_b).flatten()
                    outputs.append(pv_b)
                return torch.stack(outputs, dim=0)

            x_list = x_list.squeeze(0)
            ei_list = ei_list.squeeze(0) if ei_list.dim() == 3 else ei_list
            masks_list = masks_list.squeeze(0) if masks_list.dim() == 3 else masks_list

        # Single sample (no batch) or after squeezing batch=1
        y = x_list
        for conv in self.convs_matrix:
            y = conv(y, ei_list).relu()

        src_idx, dst_idx = ei_list
        edge_emb = (y[src_idx] + y[dst_idx]) * 0.5
        path_emb = masks_list @ edge_emb
        path_vec = self.readout_obj(path_emb)
        flat = path_vec.flatten()
        return flat.unsqueeze(0)
