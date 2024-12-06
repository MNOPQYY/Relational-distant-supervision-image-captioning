#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

"""
PyTorch modules for dealing with graphs.
"""


def _init_weights(module):
  if hasattr(module, 'weight'):
    if isinstance(module, nn.Linear):
      nn.init.kaiming_normal_(module.weight)
      
def build_mlp(dim_list, activation='leakyrelu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
  layers = []
  for i in range(len(dim_list) - 1):
    dim_in, dim_out = dim_list[i], dim_list[i + 1]
    layers.append(nn.Linear(dim_in, dim_out))
    final_layer = (i == len(dim_list) - 2)
    if not final_layer or final_nonlinearity:
      if batch_norm == 'batch':
        layers.append(nn.BatchNorm1d(dim_out))
      if activation == 'relu':
        layers.append(nn.ReLU())
      elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
  return nn.Sequential(*layers)

class GraphTripleConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, embed, emb_dim, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none'):
    super(GraphTripleConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.embed_dim = emb_dim
    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    net1_layers = [3 * emb_dim, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)
    
    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)

  def forward(self, embed, objs,obj_vecs, rels,pred_vecs, edges, rel_word_nums):
   
    bs = obj_vecs.size(0)
    O  = obj_vecs.size(1)
    T  = pred_vecs.size(1)
    Demb, Din, H, Dout =self.embed_dim, self.input_dim, self.hidden_dim, self.output_dim
    
    s_idx = edges[..., 0].contiguous() 
    o_idx = edges[..., 1].contiguous()
    obj_vecs = obj_vecs.view(-1,Din)
    ii = torch.arange(0,bs*O-1,O)  
    ii = ii.view(-1,1).expand_as(s_idx) 
    s_idx = s_idx+ii.type(torch.LongTensor).cuda()
    o_idx = o_idx+ii.type(torch.LongTensor).cuda()
    cur_s_vecs = obj_vecs[s_idx.view(-1)] 
    cur_o_vecs = obj_vecs[o_idx.view(-1)]
    cur_t_vecs_prd = pred_vecs.view(-1,Din)
    dtype, device = obj_vecs.dtype, obj_vecs.device
   
    cur_t_vecs = torch.cat([cur_s_vecs, cur_t_vecs_prd, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]
    
    new_p_vecs = new_p_vecs.view(bs, T, Dout)
 
    pooled_obj_vecs = torch.zeros(bs*O, H, dtype=dtype, device=device)
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

    if self.pooling == 'avg':
      obj_counts = torch.zeros(bs*O, dtype=dtype, device=device)
      ones = torch.ones(bs*T, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, s_idx.view(-1), ones)
      obj_counts = obj_counts.scatter_add(0, o_idx.view(-1), ones)
  
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)
      pooled_obj_vecs = pooled_obj_vecs.cuda()
    new_obj_vecs = self.net2(pooled_obj_vecs)
    new_obj_vecs = new_obj_vecs.view(bs, O, Dout)

    return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
  """ A sequence of scene graph convolution layers  """
  def __init__(self, embed, embed_dim, input_dim,output_dim, num_layers=5, hidden_dim=512, pooling='avg',
               mlp_normalization='none'):
    super(GraphTripleConvNet, self).__init__()

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    gconv_kwargs = {
      'embed': embed,
      'emb_dim': embed_dim,
      'input_dim': input_dim,
      'output_dim': output_dim,
      'hidden_dim': hidden_dim,
      'pooling': pooling,
      'mlp_normalization': mlp_normalization,
    }
    for _ in range(self.num_layers):
      self.gconvs.append(GraphTripleConv(**gconv_kwargs))

  def forward(self, embed, objs, rels, edges, rel_word_nums, n_obj, n_rels):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(embed, objs, rels, edges, rel_word_nums, n_obj, n_rels)
    return obj_vecs, pred_vecs


