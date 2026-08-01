"""Run the analytic proper D6 benzene example."""

from __future__ import annotations

import torch

from .encoder import d6_benzene_encoder
from .verification import rotation_from_rotvec


def main() -> None:
    """Print the D6 encoding and all requested numerical checks."""
    dtype = torch.float64
    encoder = d6_benzene_encoder(dtype=dtype)
    position = torch.tensor((1.2, -0.8, 0.3), dtype=dtype)
    rotation = rotation_from_rotvec(torch.tensor((0.24, -0.32, 0.17), dtype=dtype))
    rotation_code = encoder.encode_rotation(rotation)
    full_code = encoder.encode(position, rotation)
    generator_residual = max(
        float(torch.linalg.vector_norm(
            encoder.encode_rotation(rotation @ generator) - rotation_code
        ))
        for generator in encoder.generators
    )
    inverse = encoder.inverse_fiber(rotation_code, reference_rotation=rotation)
    jacobian = encoder.jacobian(rotation)
    pseudoinverse = encoder.jacobian_pseudoinverse(rotation)
    pseudoinverse_residual = float(torch.linalg.matrix_norm(
        pseudoinverse @ jacobian - torch.eye(3, dtype=dtype)
    ))

    torch.set_printoptions(precision=10, linewidth=120)
    print("group:", encoder.certificate.group_name)
    print("ranks:", encoder.ranks)
    print("global certificate:", encoder.certificate.exact)
    print("rotation encoding shape:", tuple(rotation_code.shape))
    print("rotation encoding:", rotation_code)
    print("position plus rotation shape:", tuple(full_code.shape))
    print("generator constraint residual:", generator_residual)
    print("inverse code residual:", inverse.code_residual)
    print("inverse fiber residual:", inverse.fiber_residual)
    print("original matrix inclusion error:", inverse.inclusion_error)
    print("Jacobian rank:", int(torch.linalg.matrix_rank(jacobian)))
    print("Jacobian pseudoinverse residual:", pseudoinverse_residual)
    print("PyTorch gradcheck:", encoder.verify_gradients(rotation))


if __name__ == "__main__":
    main()
