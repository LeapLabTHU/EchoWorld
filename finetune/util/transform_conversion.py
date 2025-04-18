import torch


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")



def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X", "Y", or "Z".
        angle: Tensor of Euler angles in radians.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y, or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
                    {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    
    # Vectorized calculation for batch of Euler angles
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, euler_angles.unbind(-1))]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _angle_from_tan(axis: str, other_axis: str, data: torch.Tensor, horizontal: bool, tait_bryan: bool) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of the matrix
    which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X", "Y", or "Z" for the angle we are finding.
        other_axis: Axis label "X", "Y", or "Z" for the middle axis in the
                    convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor.
    """
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    
    # Vectorized angle extraction using atan2
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2

    if tait_bryan:
        central_angle = torch.asin(matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    angles = (
        _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
        central_angle,
        _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
    )

    return torch.stack(angles, -1)


def hexa2trans(hexa: torch.Tensor) -> torch.Tensor:
    """
    Convert a 6D vector to a 4x4 transformation matrix.

    Args:
        hexa: 6D vector of translations and Euler angles in degrees.

    Returns:
        4x4 transformation matrix.
    """
    if hexa.size(-1) != 6:
        raise ValueError(f"Invalid hexa shape {hexa.shape}.")
    
    translations = hexa[..., :3]
    rotations = torch.deg2rad(hexa[..., 3:])
    rotation_matrices = euler_angles_to_matrix(rotations, "XYZ")

    transformation_matrix = torch.eye(4, device=hexa.device).expand(hexa.shape[:-1] + (4, 4)).clone()
    transformation_matrix[..., :3, :3] = rotation_matrices
    transformation_matrix[..., :3, 3] = translations

    return transformation_matrix


def trans2hexa(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert a 4x4 transformation matrix to a 6D vector.

    Args:
        matrix: 4x4 transformation matrix.

    Returns:
        6D vector of translations and Euler angles in degrees.
    """
    if matrix.size(-1) != 4 or matrix.size(-2) != 4:
        raise ValueError(f"Invalid matrix shape {matrix.shape}.")

    translations = matrix[..., :3, 3]
    rotation_matrix = matrix[..., :3, :3]
    rotations = matrix_to_euler_angles(rotation_matrix, "XYZ")
    rotations = torch.rad2deg(rotations)

    return torch.cat((translations, rotations), -1)


def hexaMatTorch(hexa1: torch.Tensor, hexa2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two 6D vectors.

    Args:
        hexa1: 6D vector of translations and Euler angles in degrees.
        hexa2: 6D vector of translations and Euler angles in degrees.
    
    Returns:
        6D vector of translations and Euler angles in degrees.
    """
    transformation1 = hexa2trans(hexa1)
    transformation2 = hexa2trans(hexa2)
    transformation = torch.matmul(transformation1, transformation2)

    return trans2hexa(transformation)


def hexaDiffTorch(hexa1: torch.Tensor, hexa2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the difference between two 6D vectors.

    Args:
        hexa1: 6D vector of translations and Euler angles in degrees.
        hexa2: 6D vector of translations and Euler angles in degrees.
    
    Returns:
        6D vector of translations and Euler angles in degrees.
    """
    transformation1 = hexa2trans(hexa1)
    transformation2 = hexa2trans(hexa2)
    transformation = torch.matmul(transformation1, torch.inverse(transformation2))

    return trans2hexa(transformation)




# PyTorch implementation of normalize_vector
def normalize_vector_torch(v):
    v_mag = torch.norm(v, dim=-1, keepdim=True)
    v_mag = torch.clamp(v_mag, min=1e-8)
    return v / v_mag

# PyTorch implementation of cross_product
def cross_product_torch(u, v):
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    
    out = torch.stack((i, j, k), dim=1)
    return out

# PyTorch implementation of compute_rotation_matrix_from_ortho6d
def compute_rotation_matrix_from_ortho6d_torch(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
    
    x = normalize_vector_torch(x_raw)
    z = cross_product_torch(x, y_raw)
    z = normalize_vector_torch(z)
    y = cross_product_torch(z, x)
    
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    
    matrix = torch.cat((x, y, z), dim=2)
    return matrix

# PyTorch implementation of compute_ortho6d_from_rotation_matrix
def compute_ortho6d_from_rotation_matrix_torch(matrix):
    ortho6d = matrix[:, :, :2].permute(0, 2, 1).reshape(matrix.shape[0], -1)
    return ortho6d


def hexa2ortho9d(hexa):
    translations = hexa[...,:3]
    rotations = torch.deg2rad(hexa[..., 3:])
    rotation_matrix = euler_angles_to_matrix(rotations, "XYZ")
    ortho6d = compute_ortho6d_from_rotation_matrix_torch(rotation_matrix)
    return torch.cat([translations, ortho6d], dim=-1)

def ortho9d2hexa(o9d):
    translations = o9d[...,:3]
    ortho6d = o9d[...,3:]
    rotation_matrix = compute_rotation_matrix_from_ortho6d_torch(ortho6d)
    rotations = matrix_to_euler_angles(rotation_matrix, "XYZ")
    rotations = torch.rad2deg(rotations)
    return torch.cat([translations, rotations], dim=-1)



    