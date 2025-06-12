import torch
from torch_geometric.nn import GATv2Conv, SAGEConv, GraphConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# TODO: (version 5.5-6) Add params to optuna

class PathGNN(BaseFeaturesExtractor):
    """
    Custom PathGNN feature extraction algorithm integrated with SB3.
    """

    def __init__(self, obs_space, emb_dim, gnn_type, layers):
        super().__init__(obs_space, features_dim=emb_dim * obs_space["path_masks"].shape[0])
        conv_map_dict = {"gat": GATv2Conv, "sage": SAGEConv, "graphconv": GraphConv}
        selected_conv_obj = conv_map_dict.get(gnn_type, GATv2Conv)
        in_dim = obs_space["x"].shape[1]
        self.convs_matrix = torch.nn.ModuleList([
            selected_conv_obj(in_dim if i == 0 else emb_dim, emb_dim)
            for i in range(layers)
        ])
        self.readout_obj = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, obs_dict: dict):
        """
        Convert observation into feature vector.
        """
        x_list = obs_dict["x"]
        edge_index = obs_dict["edge_index"].long()
        masks_list = obs_dict["path_masks"]

        if x_list.dim() == 2:
            x_list = x_list.unsqueeze(0)
            edge_index = edge_index.unsqueeze(0)
            masks_list = masks_list.unsqueeze(0)

        batch_size = x_list.size(0)
        outputs_matrix = []
        for b in range(batch_size):
            xb = x_list[b]
            eib = edge_index[b]
            mb = masks_list[b]

            y = xb
            for conv_obj in self.convs_matrix:
                y = conv_obj(y, eib).relu()

            src_idx, dst_idx = eib
            edge_emb = (y[src_idx] + y[dst_idx]) * 0.5
            path_emb = mb @ edge_emb
            pv_list = self.readout_obj(path_emb).flatten()
            outputs_matrix.append(pv_list)

        return torch.stack(outputs_matrix, dim=0)
