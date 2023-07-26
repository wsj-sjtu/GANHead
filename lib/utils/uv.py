"""
UV Operations implemented using nvdiffrast.
"""

import torch
from torch import nn
from PIL import Image
import numpy as np
import trimesh
import torch.nn.functional as F


class Uv2Attr(nn.Module):
    def __init__(self, uv_coords, faces_packed=None, size=128):
        super().__init__()
        # uv_coords 1*nvert*2
        uv_coords = uv_coords/size
        x, y = torch.chunk(uv_coords, 2, dim=2)
        x = x * (size - 1)
        y = y * (size - 1)
        self.x = x.squeeze(2)
        self.y = y.squeeze(2)
        self.faces_packed = faces_packed

    @staticmethod
    def index_vert_attr(x, y, uvmap):
        B, C, H, W = uvmap.shape
        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)
        idx = (W * y + x).type(torch.long)
        uvmap = uvmap.view(-1, C, H * W)
        idx = idx.unsqueeze(1).repeat(1, C, 1)
        vert_attr = uvmap.gather(2, idx)
        return vert_attr

    def nearest(self, uvmap):
        # not work. too much points overlap, show nothing
        x = torch.round(self.x)
        y = torch.round(self.y)
        vert_attr = self.index_vert_attr(x, y, uvmap)
        return vert_attr

    def bilinear(self, uvmap):
        x = self.x
        y = self.y
        x0 = torch.floor(x)
        x1 = x0 + 1
        y0 = torch.floor(y)
        y1 = y0 + 1
        ia = self.index_vert_attr(x0, y0, uvmap)  # 5x3x38365
        ib = self.index_vert_attr(x0, y1, uvmap)
        ic = self.index_vert_attr(x1, y0, uvmap)
        id = self.index_vert_attr(x1, y1, uvmap)

        wa = (x1 - x) * (y1 - y)  # 1x38365
        wb = (x1 - x) * (y - y0)

        wc = (x - x0) * (y1 - y)

        wd = (x - x0) * (y - y0)

        vert_attr = wa.unsqueeze(1).repeat(1, 3, 1) * ia + wb.unsqueeze(1).repeat(1, 3, 1) * ib \
                    + wc.unsqueeze(1).repeat(1, 3, 1) * ic + wd.unsqueeze(1).repeat(1, 3, 1) * id
        return vert_attr

    def get_edge_tri_idx(self, vert_attr):
        vert_packed = vert_attr.permute(0, 2, 1).reshape(-1, 3)
        faces_verts = vert_packed[self.faces_packed]
        # background vert: [0.0, 0.0, 0.0]
        mask = (faces_verts == torch.zeros_like(faces_verts[0, 0])).all(dim=2).any(dim=1)
        return mask

    def forward(self, batch_uv, bilinear=False):
        if bilinear:
            vert_attr = self.bilinear(batch_uv)
        else:
            vert_attr = self.nearest(batch_uv)
        # mask = self.get_edge_tri_idx(vert_attr)
        return vert_attr  # , mask






