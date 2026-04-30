import torch

def axis_angle_to_matrix(axis, angle):
    """
    Rodrigues' formula.
    axis: Tensor/array/list with shape (..., 3)
    angle: scalar/Tensor broadcastable to shape (...)
    return: Tensor with shape (..., 3, 3)
    """
    axis = torch.as_tensor(axis)
    angle = torch.as_tensor(angle, dtype=axis.dtype, device=axis.device)

    eps = torch.finfo(axis.dtype).eps
    norm = axis.norm(dim=-1, keepdim=True).clamp_min(eps)
    n = axis / norm  # normalize

    x, y, z = n.unbind(-1)  # (...), (...), (...)
    zeros = torch.zeros_like(x)
    K = torch.stack([
        zeros, -z,    y,
        z,     zeros, -x,
        -y,    x,     zeros
    ], dim=-1).reshape(n.shape[:-1] + (3, 3))  # (..., 3, 3)

    I = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(K.shape)
    nnt = n[..., :, None] * n[..., None, :]    # (..., 3, 3)

    c = torch.cos(angle)[..., None, None]
    s = torch.sin(angle)[..., None, None]

    R = c * I + (1.0 - c) * nnt + s * K
    return R