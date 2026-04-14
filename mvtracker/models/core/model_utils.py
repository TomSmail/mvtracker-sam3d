# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Tuple, Optional

import torch
from easydict import EasyDict as edict
from torch.nn import functional as F

from mvtracker.utils.basic import to_homogeneous, from_homogeneous

EPS = 1e-6


def smart_cat(tensor1, tensor2, dim):
    if tensor1 is None:
        return tensor2
    return torch.cat([tensor1, tensor2], dim=dim)


def normalize_single(d):
    # d is a whatever shape torch tensor
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d - dmin) / (EPS + (dmax - dmin))
    return d


def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    out = torch.zeros(d.size())
    if d.is_cuda:
        out = out.cuda()
    B = list(d.size())[0]
    for b in list(range(B)):
        out[b] = normalize_single(d[b])
    return out


def meshgrid2d(B, Y, X, stack=False, norm=False, device="cuda"):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a, b) in zip(x.size(), mask.size()):
        assert a == b  # some shape mismatch!
    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom
    return mean


def bilinear_sample2d(im, x, y, return_inbounds=False):
    # x and y are each B, N
    # output is B, C, N
    if len(im.shape) == 5:
        B, N, C, H, W = list(im.shape)
    else:
        B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)

    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64, device=x.device) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    if len(im.shape) == 5:
        im_flat = (im.permute(0, 3, 4, 1, 2)).reshape(B * H * W, N, C)
        i_y0_x0 = torch.diagonal(im_flat[idx_y0_x0.long()], dim1=1, dim2=2).permute(0, 2, 1)
        i_y0_x1 = torch.diagonal(im_flat[idx_y0_x1.long()], dim1=1, dim2=2).permute(0, 2, 1)
        i_y1_x0 = torch.diagonal(im_flat[idx_y1_x0.long()], dim1=1, dim2=2).permute(0, 2, 1)
        i_y1_x1 = torch.diagonal(im_flat[idx_y1_x1.long()], dim1=1, dim2=2).permute(0, 2, 1)
    else:
        im_flat = (im.permute(0, 2, 3, 1)).reshape(B * H * W, C)
        i_y0_x0 = im_flat[idx_y0_x0.long()]
        i_y0_x1 = im_flat[idx_y0_x1.long()]
        i_y1_x0 = im_flat[idx_y1_x0.long()]
        i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = (w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1)
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(
            B, N
        )  # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output  # B, C, N


def procrustes_analysis(X0, X1, Weight):  # [B,N,3]
    # translation
    t0 = X0.mean(dim=1, keepdim=True)
    t1 = X1.mean(dim=1, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    # s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    # s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    # X0cs = X0c/s0
    # X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U, _, V = (X0c.t() @ X1c).double().svd(some=True)
    R = (U @ V.t()).float()
    if R.det() < 0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/@R.t()+t0
    se3 = edict(t0=t0[0], t1=t1[0], R=R)

    return se3


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
        )
    else:
        coords = coords * torch.tensor([2 / size for size in reversed(sizes)], device=coords.device)

    coords -= 1

    return F.grid_sample(input, coords, align_corners=align_corners, padding_mode=padding_mode)


def sample_features4d(input, coords):
    r"""Sample spatial features

    `sample_features4d(input, coords)` samples the spatial features
    :attr:`input` represented by a 4D tensor :math:`(B, C, H, W)`.

    The field is sampled at coordinates :attr:`coords` using bilinear
    interpolation. :attr:`coords` is assumed to be of shape :math:`(B, R,
    3)`, where each sample has the format :math:`(x_i, y_i)`. This uses the
    same convention as :func:`bilinear_sampler` with `align_corners=True`.

    The output tensor has one feature per point, and has shape :math:`(B,
    R, C)`.

    Args:
        input (Tensor): spatial features.
        coords (Tensor): points.

    Returns:
        Tensor: sampled features.
    """

    B, _, _, _ = input.shape

    # B R 2 -> B R 1 2
    coords = coords.unsqueeze(2)

    # B C R 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 1, 3).view(
        B, -1, feats.shape[1] * feats.shape[3]
    )  # B C R 1 -> B R C


def sample_features5d(input, coords):
    r"""Sample spatio-temporal features

    `sample_features5d(input, coords)` works in the same way as
    :func:`sample_features4d` but for spatio-temporal features and points:
    :attr:`input` is a 5D tensor :math:`(B, T, C, H, W)`, :attr:`coords` is
    a :math:`(B, R1, R2, 3)` tensor of spatio-temporal point :math:`(t_i,
    x_i, y_i)`. The output tensor has shape :math:`(B, R1, R2, C)`.

    Args:
        input (Tensor): spatio-temporal features.
        coords (Tensor): spatio-temporal points.

    Returns:
        Tensor: sampled features.
    """

    B, T, _, _, _ = input.shape

    # B T C H W -> B C T H W
    input = input.permute(0, 2, 1, 3, 4)

    # B R1 R2 3 -> B R1 R2 1 3
    coords = coords.unsqueeze(3)

    # B C R1 R2 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 3, 1, 4).view(
        B, feats.shape[2], feats.shape[3], feats.shape[1]
    )  # B C R1 R2 1 -> B R1 R2 C


def pixel_xy_and_camera_z_to_world_space(pixel_xy, camera_z, intrs_inv, extrs_inv):
    num_frames, num_points, _ = pixel_xy.shape
    assert pixel_xy.shape == (num_frames, num_points, 2)
    assert camera_z.shape == (num_frames, num_points, 1)
    assert intrs_inv.shape == (num_frames, 3, 3)
    assert extrs_inv.shape == (num_frames, 4, 4)

    pixel_xy_homo = torch.cat([pixel_xy, pixel_xy.new_ones(pixel_xy[..., :1].shape)], -1)
    camera_xyz = torch.einsum('Aij,ABj->ABi', intrs_inv, pixel_xy_homo) * camera_z
    camera_xyz_homo = torch.cat([camera_xyz, camera_xyz.new_ones(camera_xyz[..., :1].shape)], -1)
    world_xyz_homo = torch.einsum('Aij,ABj->ABi', extrs_inv, camera_xyz_homo)
    if not torch.allclose(
            world_xyz_homo[..., -1],
            world_xyz_homo.new_ones(world_xyz_homo[..., -1].shape),
            atol=0.1,
    ):
        warnings.warn(f"pixel_xy_and_camera_z_to_world_space found some homo coordinates not close to 1: "
                      f"the homo values are in {world_xyz_homo[..., -1].min()} – {world_xyz_homo[..., -1].max()}")
    world_xyz = world_xyz_homo[..., :-1]

    assert world_xyz.shape == (num_frames, num_points, 3)
    return world_xyz


def world_space_to_pixel_xy_and_camera_z(world_xyz, intrs, extrs):
    num_frames, num_points, _ = world_xyz.shape
    assert world_xyz.shape == (num_frames, num_points, 3)
    assert intrs.shape == (num_frames, 3, 3)
    assert extrs.shape == (num_frames, 3, 4)

    world_xyz_homo = torch.cat([world_xyz, world_xyz.new_ones(world_xyz[..., :1].shape)], -1)
    camera_xyz = torch.einsum('Aij,ABj->ABi', extrs, world_xyz_homo)
    camera_z = camera_xyz[..., -1:]
    pixel_xy_homo = torch.einsum('Aij,ABj->ABi', intrs, camera_xyz)
    pixel_xy = pixel_xy_homo[..., :2] / pixel_xy_homo[..., -1:]

    assert pixel_xy.shape == (num_frames, num_points, 2)
    assert camera_z.shape == (num_frames, num_points, 1)
    return pixel_xy, camera_z


def get_points_on_a_grid(
        size: int,
        extent: Tuple[float, ...],
        center: Optional[Tuple[float, ...]] = None,
        device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def init_pointcloud_from_rgbd(
        fmaps: torch.Tensor,
        depths: torch.Tensor,
        intrs: torch.Tensor,
        extrs: torch.Tensor,
        stride=4,
        level=0,
        depth_interp_mode='nearest',
        return_validity_mask=False,
):
    B, V, S, C, H, W = fmaps.shape
    assert fmaps.shape == (B, V, S, C, H, W)
    assert depths.shape == (B, V, S, 1, H, W)
    assert intrs.shape == (B, V, S, 3, 3)
    assert extrs.shape == (B, V, S, 3, 4)

    # Pool the fmaps and depths to the desired pyramid level
    fmaps = fmaps.reshape(B * V * S, C, H, W)
    depths = depths.reshape(B * V * S, 1, H, W)
    for i in range(level):
        fmaps = F.avg_pool2d(fmaps, 2, stride=2)
        if depth_interp_mode == 'avg':
            depths = F.avg_pool2d(depths, 2, stride=2)
        elif depth_interp_mode == 'nearest':
            depths = F.interpolate(depths, scale_factor=0.5, mode='nearest')
        else:
            raise NotImplementedError
    H = H // 2 ** level
    W = W // 2 ** level
    fmaps = fmaps.reshape(B, V, S, C, H, W)
    depths = depths.reshape(B, V, S, 1, H, W)
    stride = stride * 2 ** level

    # Invert intrinsics and extrinsics
    intrs_inv = torch.inverse(intrs.float()).type(intrs.dtype)
    extrs_square = torch.eye(4).to(extrs.device)[None].repeat(B, V, S, 1, 1)
    extrs_square[:, :, :, :3, :] = extrs
    extrs_inv = torch.inverse(extrs_square.float()).type(extrs.dtype)
    assert intrs_inv.shape == (B, V, S, 3, 3)
    assert extrs_inv.shape == (B, V, S, 4, 4)

    # Pixel --> Camera --> World
    pixel_xy = torch.stack(torch.meshgrid(
        (torch.arange(0, H) + 0.5) * stride - 0.5,
        (torch.arange(0, W) + 0.5) * stride - 0.5,
        indexing="ij",
    )[::-1], dim=-1)
    pixel_xy = pixel_xy.to(device=fmaps.device, dtype=fmaps.dtype)
    pixel_xy_homo = to_homogeneous(pixel_xy)
    depthmap_camera_xyz = torch.einsum('BVSij,HWj->BVSHWi', intrs_inv, pixel_xy_homo)
    depthmap_camera_xyz = depthmap_camera_xyz * depths[..., 0, :, :, None]
    depthmap_camera_xyz_homo = to_homogeneous(depthmap_camera_xyz)
    depthmap_world_xyz_homo = torch.einsum('BVSij,BVSHWj->BVSHWi', extrs_inv, depthmap_camera_xyz_homo)
    depthmap_world_xyz = from_homogeneous(depthmap_world_xyz_homo)

    pointcloud_xyz = depthmap_world_xyz.permute(0, 2, 1, 3, 4, 5).reshape(B * S, V * H * W, 3)
    pointcloud_fvec = fmaps.permute(0, 2, 1, 4, 5, 3).reshape(B * S, V * H * W, C)

    if return_validity_mask:
        pointcloud_valid_mask = depths.permute(0, 2, 1, 3, 4, 5).reshape(B * S, V * H * W) > 0
        return pointcloud_xyz, pointcloud_fvec, pointcloud_valid_mask

    return pointcloud_xyz, pointcloud_fvec


def augment_pointcloud_with_mesh_vertices(
        pointcloud_xyz: torch.Tensor,
        pointcloud_fvec: torch.Tensor,
        mesh_vertices: torch.Tensor,
        fmaps: torch.Tensor,
        intrs: torch.Tensor,
        extrs: torch.Tensor,
        stride: int,
        level: int = 0,
        pointcloud_valid: Optional[torch.Tensor] = None,
        depths: Optional[torch.Tensor] = None,
        fusion_mode: str = 'mean',
        alpha_depth: float = 15.0,
        alpha_feat: float = 5.0,
        min_confidence: float = 0.0,
) -> tuple:
    """
    Augment the depth-based point cloud with SAM3D mesh vertices.

    Projects mesh vertices into each view's feature map, bilinearly samples features,
    and fuses across views using the specified fusion mode. Concatenates the result to
    the existing point cloud.

    Args:
        pointcloud_xyz:  (B*S, V*H*W, 3)
        pointcloud_fvec: (B*S, V*H*W, C)
        mesh_vertices:   (B*S, M, 3) world-space mesh vertices
        fmaps:           (B, V, S, C, H, W) original feature maps (before pyramid pooling)
        intrs:           (B, V, S, 3, 3)
        extrs:           (B, V, S, 3, 4) world-to-camera
        stride:          base pixel stride (before pyramid scaling)
        level:           pyramid level (fmaps are pooled by 2^level)
        pointcloud_valid: (B*S, V*H*W) optional validity mask
        depths:          (B, V, S, 1, H, W) depth maps for depth-consistency weighting
        fusion_mode:     'mean' (simple average), 'depth_weighted', or 'consensus'
        alpha_depth:     sharpness for depth-consistency weighting
        alpha_feat:      sharpness for feature consensus weighting
        min_confidence:  filter vertices with max view weight below this threshold

    Returns:
        Augmented (pointcloud_xyz, pointcloud_fvec) and optionally pointcloud_valid.
    """
    B, V, S, C, H_orig, W_orig = fmaps.shape

    # Pool fmaps to the correct pyramid level
    if level > 0:
        fmaps_flat = fmaps.reshape(B * V * S, C, H_orig, W_orig)
        for _ in range(level):
            fmaps_flat = F.avg_pool2d(fmaps_flat, 2, stride=2)
        H = H_orig // 2 ** level
        W = W_orig // 2 ** level
        fmaps = fmaps_flat.reshape(B, V, S, C, H, W)
    else:
        H, W = H_orig, W_orig

    # Pool depths to the same pyramid level (for depth-consistency weighting)
    if depths is not None and fusion_mode in ('depth_weighted', 'consensus'):
        if level > 0:
            depths_flat = depths.reshape(B * V * S, 1, depths.shape[-2], depths.shape[-1])
            for _ in range(level):
                depths_flat = F.avg_pool2d(depths_flat, 2, stride=2)
            depths_pooled = depths_flat.reshape(B, V, S, 1, H, W)
        else:
            depths_pooled = depths
    else:
        depths_pooled = None

    effective_stride = stride * 2 ** level
    BS = B * S
    M = mesh_vertices.shape[1]
    device = mesh_vertices.device
    use_depth_weighting = fusion_mode in ('depth_weighted', 'consensus') and depths_pooled is not None
    use_consensus = fusion_mode == 'consensus' and depths_pooled is not None

    # Accumulate features across views via weighted mean
    feat_sum = torch.zeros(BS, M, C, device=device, dtype=fmaps.dtype)
    weight_sum = torch.zeros(BS, M, 1, device=device, dtype=fmaps.dtype)

    # For consensus mode, store per-view data for the second pass
    if use_consensus:
        per_view_feats = []
        per_view_weights = []

    for v in range(V):
        # Get per-frame intrinsics and extrinsics for this view: (B, S, ...)
        intrs_v = intrs[:, v]  # (B, S, 3, 3)
        extrs_v = extrs[:, v]  # (B, S, 3, 4)

        # Reshape to (B*S, ...)
        intrs_v_flat = intrs_v.reshape(BS, 3, 3)
        extrs_v_flat = extrs_v.reshape(BS, 3, 4)

        # Project: world -> camera -> pixel
        # mesh_vertices: (BS, M, 3)
        world_homo = torch.cat([mesh_vertices, mesh_vertices.new_ones(BS, M, 1)], dim=-1)  # (BS, M, 4)
        cam_xyz = torch.einsum('bij,bmj->bmi', extrs_v_flat, world_homo)  # (BS, M, 3)
        cam_z = cam_xyz[..., 2:3]  # (BS, M, 1)

        pixel_homo = torch.einsum('bij,bmj->bmi', intrs_v_flat, cam_xyz)  # (BS, M, 3)
        pixel_xy = pixel_homo[..., :2] / (pixel_homo[..., 2:3] + 1e-8)  # (BS, M, 2)

        # Visibility: z > 0 and within image bounds (in original pixel space)
        px = pixel_xy[..., 0]
        py = pixel_xy[..., 1]
        H_px = H * effective_stride
        W_px = W * effective_stride
        visible = (cam_z[..., 0] > 0) & (px >= 0) & (px < W_px) & (py >= 0) & (py < H_px)  # (BS, M)

        # Convert to feature map coordinates
        feat_x = pixel_xy[..., 0] / effective_stride  # (BS, M)
        feat_y = pixel_xy[..., 1] / effective_stride  # (BS, M)

        # Get the feature map for this view: (B, S, C, H, W) -> (BS, C, H, W)
        fmap_v = fmaps[:, v].reshape(BS, C, H, W)

        # Bilinear sample: expects (B, C, H, W) and x, y each (B, N) -> output (B, C, N)
        sampled = bilinear_sample2d(fmap_v, feat_x, feat_y)  # (BS, C, M)
        sampled = sampled.permute(0, 2, 1)  # (BS, M, C)

        vis_mask = visible.unsqueeze(-1).float()  # (BS, M, 1)

        # Compute per-view weight
        if use_depth_weighting:
            # Sample depth map at projected pixel location
            dmap_v = depths_pooled[:, v].reshape(BS, 1, H, W)  # (BS, 1, H, W)
            sampled_depth = bilinear_sample2d(dmap_v, feat_x, feat_y)  # (BS, 1, M)
            sampled_depth = sampled_depth.permute(0, 2, 1)  # (BS, M, 1)

            # Depth consistency: ratio of depth map value to projected vertex depth
            # ratio ~ 1.0 means vertex is at the visible surface; ratio << 1 means occluded
            depth_ratio = sampled_depth / (cam_z.abs() + 1e-8)
            # Also ignore where depth map is zero/invalid
            depth_valid = (sampled_depth > 1e-4).float()
            w_depth = torch.exp(-alpha_depth * (1.0 - depth_ratio).abs()) * depth_valid
            # For invalid depth, fall back to binary visibility (weight=1)
            w_depth = w_depth + (1.0 - depth_valid)

            view_weight = vis_mask * w_depth  # (BS, M, 1)
        else:
            view_weight = vis_mask  # (BS, M, 1)

        # Accumulate
        feat_sum = feat_sum + sampled * view_weight
        weight_sum = weight_sum + view_weight

        if use_consensus:
            per_view_feats.append(sampled)
            per_view_weights.append(view_weight)

    if use_consensus and len(per_view_feats) > 0:
        # Two-pass fusion: compute warm mean, then re-weight by feature consensus
        warm_mean = feat_sum / (weight_sum + 1e-8)  # (BS, M, C)

        # Second pass: re-accumulate with consensus weighting
        feat_sum_refined = torch.zeros_like(feat_sum)
        weight_sum_refined = torch.zeros_like(weight_sum)

        for v in range(V):
            feat_v = per_view_feats[v]  # (BS, M, C)
            w_v = per_view_weights[v]  # (BS, M, 1)

            # Cosine similarity between this view's feature and the warm mean
            cos_sim = F.cosine_similarity(feat_v, warm_mean, dim=-1).unsqueeze(-1)  # (BS, M, 1)
            w_consensus = torch.exp(alpha_feat * cos_sim)  # higher similarity = higher weight

            w_total = w_v * w_consensus
            feat_sum_refined = feat_sum_refined + feat_v * w_total
            weight_sum_refined = weight_sum_refined + w_total

        mesh_fvec = feat_sum_refined / (weight_sum_refined + 1e-8)
    else:
        # Simple weighted (or unweighted) mean
        mesh_fvec = feat_sum / (weight_sum + 1e-8)  # (BS, M, C)

    # Optional: filter low-confidence vertices
    if min_confidence > 0.0:
        max_weight = weight_sum.squeeze(-1)  # (BS, M)
        confident = max_weight > min_confidence
        # Keep all vertices but zero out features for low-confidence ones
        mesh_fvec = mesh_fvec * confident.unsqueeze(-1).float()

    # Concatenate to existing point cloud
    aug_xyz = torch.cat([pointcloud_xyz, mesh_vertices], dim=1)
    aug_fvec = torch.cat([pointcloud_fvec, mesh_fvec], dim=1)

    if pointcloud_valid is not None:
        mesh_valid = torch.ones(BS, M, device=device, dtype=pointcloud_valid.dtype)
        aug_valid = torch.cat([pointcloud_valid, mesh_valid], dim=1)
        return aug_xyz, aug_fvec, aug_valid

    return aug_xyz, aug_fvec


def save_pointcloud_to_ply(filename, points, colors, edges=None):
    with open(filename, 'w') as ply_file:
        ply_file.write("ply\nformat ascii 1.0\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\nproperty float y\nproperty float z\n")
        ply_file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")

        if edges is not None:
            ply_file.write(f"element edge {len(edges)}\n")
            ply_file.write("property int vertex1\nproperty int vertex2\n")

        ply_file.write("end_header\n")

        # Write vertices (points with colors)
        for point, color in zip(points, colors):
            ply_file.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

        # Write edges (if provided)
        if edges is not None:
            for edge in edges:
                ply_file.write(f"{edge[0]} {edge[1]}\n")
