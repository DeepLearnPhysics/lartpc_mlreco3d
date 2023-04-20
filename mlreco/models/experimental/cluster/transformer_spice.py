import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me

from mlreco.models.layers.common.uresnet_layers import UResNetDecoder, UResNetEncoder
from mlreco.models.experimental.transformers.positional_encodings import FourierEmbeddings
# from mlreco.models.experimental.cluster.pointnet2.pointnet2_utils import furthest_point_sample
from mlreco.models.experimental.transformers.positional_encodings import get_normalized_coordinates
from mlreco.utils.globals import *
from mlreco.models.experimental.transformers.transformer import GenericMLP

class QueryModule(nn.Module):

    def __init__(self, cfg, name='query_module'):
        super(QueryModule, self).__init__()
    
        self.model_config = cfg[name]

        # Define instance query modules
        self.num_input = self.model_config.get('num_input', 32)
        self.num_pos_input = self.model_config.get('num_pos_input', 128)
        self.num_queries = self.model_config.get('num_queries', 200)
        # self.num_classes = self.model_config.get('num_classes', 5)
        self.mask_dim = self.model_config.get('mask_dim', 128)
        self.query_type = self.model_config.get('query_type', 'fps')
        self.query_proj = None

        if self.query_type == 'fps':
            self.query_projection = GenericMLP(
                input_dim=self.num_input,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                norm_fn_name='bn1d',
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True
            )
            self.query_pos_projection = GenericMLP(
                input_dim=self.num_pos_input,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                norm_fn_name='bn1d',
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True
            )
        elif self.query_type == 'embedding':
            self.query_feat = nn.Embedding(self.num_queries, self.mask_dim)
            self.query_pos  = nn.Embedding(self.num_queries, self.mask_dim)
        else:
            raise ValueError("Query type {} is not supported!".format(self.query_type))

        self.pos_enc = FourierEmbeddings(cfg)

    def forward(self, x, uresnet_features):
        '''
        Inputs
        ------
            x: Input ME.SparseTensor from UResNet output
        '''

        batch_size = len(x.decomposed_coordinates)

        if self.query_type == 'fps':
            # Sample query points via FPS
            fps_idx = None
            # fps_idx = [furthest_point_sample(x.decomposed_coordinates[i][None, ...].float(), 
            #            self.num_queries).squeeze(0).long() \
            #    for i in range(len(x.decomposed_coordinates))]
            # B, nqueries, 3
            sampled_coords = torch.stack([x.decomposed_coordinates[i][fps_idx[i], :] \
                                          for i in range(len(x.decomposed_coordinates))], axis=0)
            query_pos = self.pos_enc(sampled_coords.float()).permute(0, 2, 1) # B, dim, nqueries
            query_pos = self.query_pos_projection(query_pos) # B, dim, mask_dim
            queries = torch.stack([uresnet_features.decomposed_features[i][fps_idx[i].long(), :] \
                                   for i in range(len(fps_idx))]) # B, nqueries, num_uresnet_feats
            queries = queries.permute(0, 2, 1) # B, num_uresnet_feats, nqueries
            queries = self.query_projection(queries) # B, mask_dim, nqueries
        elif self.query_type == 'embedding':
            queries = self.query_feat.weight.unsqueze(0).repeat(batch_size, 1, 1)
            query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)
        else:
            raise ValueError("Query type {} is not supported!".format(self.query_type))
        
        return queries.permute((0, 2, 1)), query_pos.permute((0, 2, 1)), fps_idx
        

class TransformerSPICE(nn.Module):
    """
    Transformer based model for particle clustering, using Mask3D
    as a backbone.
    
    Mask3D backbone implementation: https://github.com/JonasSchult/Mask3D
    
    Mask3D: https://arxiv.org/abs/2210.03105
    
    """
    
    def __init__(self, cfg, name='mask3d'):
        super(TransformerSPICE, self).__init__()
        
        self.model_config = cfg[name]
        
        self.encoder = UResNetEncoder(cfg, name='uresnet')
        self.decoder = UResNetDecoder(cfg, name='uresnet')
        
        num_params_backbone = sum(p.numel() for p in self.encoder.parameters())
        num_params_backbone += sum(p.numel() for p in self.decoder.parameters())
        print(f"Number of Backbone Parameters = {num_params_backbone}")

        self.query_module = QueryModule(cfg)
        
        num_features     = self.encoder.num_filters
        self.D           = self.model_config.get('D', 3)
        self.mask_dim    = self.model_config.get('mask_dim', 128)
        self.num_classes = self.model_config.get('num_classes', 2)
        self.num_heads   = self.model_config.get('num_heads', 8)
        self.dropout     = self.model_config.get('dropout', 0.0)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.spatial_size = self.model_config.get('spatial_size', [2753, 1056, 5966])
        self.spatial_size = torch.Tensor(self.spatial_size).float().to(device)

        self.depth        = self.model_config.get('depth', 5)
        self.mask_head    = ME.MinkowskiConvolution(num_features, 
                                                    self.mask_dim, 
                                                    kernel_size=1, 
                                                    stride=1, 
                                                    bias=True, 
                                                    dimension=self.D)
        self.pooling      = ME.MinkowskiAvgPooling(kernel_size=2, 
                                                   stride=2, 
                                                   dimension=3)
        self.adc_to_mev   = 1./350.
        
        # Query Refinement Modules
        self.num_transformers = self.model_config.get('num_transformers', 3)
        self.shared_decoders = self.model_config.get('shared_decoders', False)
        
        self.instance_to_mask = nn.Linear(self.mask_dim, self.mask_dim)
        self.instance_to_class = nn.Linear(self.mask_dim, self.mask_dim)

        # Layerwise Projections
        self.linear_squeeze = nn.ModuleList()
        for i in range(self.depth-1, 0, -1):
            self.linear_squeeze.append(nn.Linear(i * num_features,
                                                 self.mask_dim))

        # Transformer Modules
        if self.shared_decoders:
            num_shared = 1
        else:
            num_shared = self.num_transformers

        self.transformers = []
        
        for num_trans in range(num_shared):
            self.transformers.append(nn.TransformerDecoderLayer(
                self.mask_dim, self.num_heads, dim_feedforward=1024, batch_first=True))
            
        self.transformers = nn.ModuleList(self.transformers)
        self.layernorm    = nn.LayerNorm(self.mask_dim)

        self.sample_sizes = [200, 800, 1600, 6400, 12800]
        

    def mask_module(self, queries, mask_features, 
                    return_attention_mask=True,
                    num_pooling_steps=0):
        '''
        Inputs
        ------
            - queries: [B, num_queries, query_dim] torch.Tensor
            - mask_features: ME.SparseTensor from mask head output
        '''
        query_feats = self.layernorm(queries)
        mask_embed = self.instance_to_mask(query_feats)
        output_class = self.instance_to_class(query_feats)

        output_masks = []

        coords, feats = mask_features.decomposed_coordinates_and_features
        batch_size = len(coords)

        assert mask_embed.shape[0] == batch_size

        for i in range(len(mask_features.decomposed_features)):
            mask = feats[i] @ mask_embed[i].T
            output_masks.append(mask)

        output_masks = torch.cat(output_masks, dim=0)
        output_coords = torch.cat(coords, dim=0)
        output_mask = me.SparseTensor(features=output_masks,
                                      coordinate_manager=mask_features.coordinate_manager,
                                      coordinate_map_key=mask_features.coordinate_map_key)
        
        if return_attention_mask:
            # nn.MultiHeadAttention attn_mask prevents "True" pixels from access
            # Hence the < 0.5 in the attn_mask
            with torch.no_grad():
                attn_mask = output_mask
                for _ in range(num_pooling_steps):
                    attn_mask = self.pooling(attn_mask.float())
                attn_mask = me.SparseTensor(features=(attn_mask.F.detach().sigmoid() < 0.5),
                                            coordinate_manager=attn_mask.coordinate_manager,
                                            coordinate_map_key=attn_mask.coordinate_map_key)
                return output_mask, output_class, attn_mask
        else:
            return output_mask, output_class
        

    def sampling_module(self, decomposed_feats, decomposed_coords, decomposed_attn, depth, 
                        max_sample_size=False, is_eval=False):
        
        indices, masks = [], []

        if min([pcd.shape[0] for pcd in decomposed_feats]) == 1:
            raise RuntimeError("only a single point gives nans in cross-attention")

        decomposed_pos_encs = []

        for coords in decomposed_coords:
            pos_enc = self.query_module.pos_enc(coords.float())
            decomposed_pos_encs.append(pos_enc)

        device = decomposed_feats[0].device

        curr_sample_size = max([pcd.shape[0] for pcd in decomposed_feats])
        if not (max_sample_size or is_eval):
            curr_sample_size = min(curr_sample_size, self.sample_sizes[depth])

        for bidx in range(len(decomposed_feats)):
            num_points = decomposed_feats[bidx].shape[0]
            if num_points <= curr_sample_size:
                idx = torch.zeros(curr_sample_size,
                                    dtype=torch.long,
                                    device=device)

                midx = torch.ones(curr_sample_size,
                                    dtype=torch.bool,
                                    device=device)

                idx[:num_points] = torch.arange(num_points,
                                              device=device)

                midx[:num_points] = False  # attend to first points
            else:
                # we have more points in pcd as we like to sample
                # take a subset (no padding or masking needed)
                idx = torch.randperm(decomposed_feats[bidx].shape[0],
                                     device=device)[:curr_sample_size]
                midx = torch.zeros(curr_sample_size,
                                   dtype=torch.bool,
                                   device=device)  # attend to all
            indices.append(idx)
            masks.append(midx)

        batched_feats = torch.stack([
            decomposed_feats[b][indices[b], :] for b in range(len(indices))
        ])
        batched_attn = torch.stack([
            decomposed_attn[b][indices[b], :] for b in range(len(indices))
        ])
        batched_pos_enc = torch.stack([
            decomposed_pos_encs[b][indices[b], :] for b in range(len(indices))
        ])

        # Mask to handle points less than num_sample points
        m = torch.stack(masks)
        # If sum(1) == nsamples, then this query has no active voxels
        batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == indices[0].shape[0]] = False
        # Fianl attention map is intersection of attention map and
        # valid voxel samples (m). 
        batched_attn = torch.logical_or(batched_attn, m[..., None])

        return batched_feats, batched_attn, batched_pos_enc
        
        
    def forward(self, point_cloud):
        
        coords = point_cloud[:, COORD_COLS].int()
        feats = point_cloud[:, VALUE_COL].float().view(-1, 1)

        normed_coords = get_normalized_coordinates(coords, self.spatial_size)
        normed_feats = feats * self.adc_to_mev 
        features = torch.cat([normed_coords, normed_feats], dim=1)
        x = ME.SparseTensor(coordinates=point_cloud[:, :VALUE_COL].int(), 
                            features=features)
        encoderOutput = self.encoder(x)
        decoderOutput = self.decoder(encoderOutput['finalTensor'], 
                                     encoderOutput['encoderTensors'])
        queries, query_pos, query_index = self.query_module(x, decoderOutput[-1])
        
        total_num_pooling = len(decoderOutput)-1
        full_res_fmap = decoderOutput[-1]
        mask_features = self.mask_head(full_res_fmap)
        batch_size = int(torch.unique(x.C[:, 0]).shape[0])
        
        predictions_mask = []
        predictions_class = []
        
        for tf_index in range(self.num_transformers):
            if self.shared_decoders:
                transformer_index = 0
            else:
                transformer_index = tf_index
            for i, fmap in enumerate(decoderOutput):
                assert queries.shape == (batch_size, 
                                         self.query_module.num_queries, 
                                         self.mask_dim)
                num_pooling = total_num_pooling-i
                
                output_mask, output_class, attn_mask = self.mask_module(queries, 
                                                          mask_features, 
                                                          num_pooling_steps=num_pooling)
                
                predictions_mask.append(output_mask.F)
                predictions_class.append(output_class)
                
                fmaps, attn_masks = fmap.decomposed_features, attn_mask.decomposed_features
                decomposed_coords = fmap.decomposed_coordinates
                
                batched_feats, batched_attn, batched_pos_enc = self.sampling_module(
                    fmaps, decomposed_coords, attn_masks, i)

                src_pcd = self.linear_squeeze[i](batched_feats)
                
                batched_attn = torch.repeat_interleave(batched_attn.permute((0, 2, 1)), repeats=8, dim=0)
                
                output = self.transformers[transformer_index](queries + query_pos,
                                                              src_pcd + batched_pos_enc)
                                                            #   memory_mask=batched_attn)
                
                queries = output
                
        output_mask, output_class, attn_mask = self.mask_module(queries, 
                                                  mask_features,
                                                  return_attention_mask=True,
                                                  num_pooling_steps=0)

        res = {
            'pred_masks' :  [output_mask.F],
            'pred_logits': [output_class],
            'aux_masks': [predictions_mask],
            'aux_classes': [predictions_class],
            'query_index': [query_index]
        }
        
        return res