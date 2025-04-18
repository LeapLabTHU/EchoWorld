import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Union, List


class Rotation:
    """
    Rotation class for calculating rotation matrix, Euler angles and quaternion.
    """
    def __init__(self) -> None:
        pass
    
    @classmethod
    def rotate(cls,
               theta: float,
               axis: str,
               degrees: bool = True) -> np.ndarray:
        """
        calculate the rotation matrix (3x3) for a given axis (X, Y, or Z) and angle 
        
        Parameters:
            theta (float): The rotation angle
            axis (str): 'x', 'y', or 'z' axis
            degrees (bool): whether the input rotation angle in degrees, default is true.
    
        Returns:
            np.array: The 3x3 rotation matrix.
        """
        # Convert angle from degrees to radians
        if degrees:
            theta = np.radians(theta)
        
        if axis == 'x':
            rotation_matrix = np.array([[1, 0, 0],
                                        [0, np.cos(theta), -np.sin(theta)],
                                        [0, np.sin(theta), np.cos(theta)]])
        elif axis == 'y':
            rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                        [0, 1, 0],
                                        [-np.sin(theta), 0, np.cos(theta)]])
        elif axis == 'z':
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
        else:
            raise ValueError("The axis must be 'x', 'y', or 'z'")
        return rotation_matrix

    @classmethod
    def euler2quat(cls,
                   angles:Union[np.ndarray, List[float]], 
                   order:str='xyz',
                   degrees:bool=False,
                   flip:bool=False) -> np.ndarray:
        """Convert Euler angles to quaternion.
    
        Args:
            angles (np.ndarray): Euler angles in radians or degrees.
            order (str): Order of Euler angles. Default: 'xyz'.
            degrees (bool): Whether the input angles are in degrees. Default: False.
            :param flip: whether to flip the order of quaternion, default: False
                the format of the output quaternion is (x, y, z, w). if (w, x, y, z) is wanted, then flip=True
        Returns:
            np.ndarray: Quaternion.
        """
        # Convert to numpy array
        if isinstance(angles, list):
            angles = np.array(angles)
        # Convert to quaternion
        r = R.from_euler(order, angles, degrees=degrees)
        q = r.as_quat()
        # Flip quaternion
        if flip:
            q = np.roll(q, 1)
        return q

    @classmethod
    def quat2euler(cls, q:Union[np.ndarray, List[float]], 
                   order:str='xyz',
                   degrees:bool=False,
                   flip:bool=False) -> np.ndarray:
        """Convert quaternion to Euler angles.
    
        Args:
            q (np.ndarray): Quaternion in (x, y, z, w) format.
            order (str): Order of Euler angles. Default: 'xyz'.
            degrees (bool): Whether to return the angles in degrees. Default: False.
            flip (bool): Whether to flip the order of quaternion. Default: False.
                if the format of q is (w, x, y, z), then flip=True
        Returns:
            np.ndarray: Euler angles in radians.
        """
        # Convert to numpy array
        if isinstance(q, list):
            q = np.array(q)
        # Flip quaternion
        if flip:
            q = np.roll(q, -1)
        # Convert to Euler angles
        r = R.from_quat(q)
        angles = r.as_euler(order, degrees=degrees)
        return angles

    @classmethod
    def euler2mat(cls, angles:Union[np.ndarray, List[float]],
                  order:str='xyz',
                  degrees:bool=False) -> np.ndarray:
        """Convert Euler angles to rotation matrix.
    
        Args:
            angles (np.ndarray): Euler angles in radians or degrees.
            order (str): Order of Euler angles. Default: 'xyz'.
            degrees (bool): Whether the input angles are in degrees. Default: True.
        Returns:
            np.ndarray: Rotation matrix.
        """
        # Convert to numpy array
        if isinstance(angles, list):
            angles = np.array(angles)
        # Convert to rotation matrix
        r = R.from_euler(order, angles, degrees=degrees)
        mat = r.as_matrix()
        return mat

    @classmethod
    def mat2euler(cls, mat:Union[np.ndarray, List[float]], 
                  order:str='xyz',
                  degrees:bool=False) -> np.ndarray:
        """Convert rotation matrix to euler angles.
    
        Args:
            mat (np.ndarray): Rotation matrix.
            order (str): Order of Euler angles. Default: 'xyz'.
            degrees (bool): Whether the input angles are in degrees. Default: True.
        Returns:
            np.ndarray: Euler angles in radians.
        """
        # Convert to numpy array
        if isinstance(mat, list):
            mat = np.array(mat)
        # Convert to Euler angles
        r = R.from_matrix(mat)
        angles = r.as_euler(order, degrees=degrees)
        return angles

    @classmethod
    def quat2mat(cls, q:Union[np.ndarray, List[float]], 
                 flip:bool=False) -> np.ndarray:
        """Convert quaternion to rotation matrix.
    
        Args:
            q (np.ndarray): Quaternion in format (x, y, z, w).
            flip (bool): Whether to flip the order of quaternion. Default: False.
                if the format of q is (w, x, y, z), then flip=True
        Returns:
            np.ndarray: Rotation matrix.
        """
        # Convert to numpy array
        if isinstance(q, list):
            q = np.array(q)
        # Flip quaternion
        if flip:
            q = np.roll(q, -1)
        # Convert to rotation matrix
        r = R.from_quat(q)
        mat = r.as_matrix()
        return mat

    @classmethod
    def mat2quat(cls, mat:Union[np.ndarray, List[float]], 
                 flip:bool=False) -> np.ndarray:
        """Convert rotation matrix to quaternion.
    
        Args:
            mat (np.ndarray): Rotation matrix.
            flip (bool): Whether to flip the order of quaternion. Default: False.
                as default, the format of the output quaternion is (x, y, z, w).
                if the format (w, x, y, z) is wanted, then flip=True
        Returns:
            np.ndarray: Quaternion.
        """
        # Convert to numpy array
        if isinstance(mat, list):
            mat = np.array(mat)
        # Convert to quaternion
        r = R.from_matrix(mat)
        q = r.as_quat()
        # Flip quaternion
        if flip:
            q = np.roll(q, 1)
        return q

    @classmethod
    def quat_diff(cls, quat_curr:Union[np.ndarray, List[float]], 
                 quat_des:Union[np.ndarray, List[float]],
                 flip:bool=False) -> np.ndarray:
        """Calculate the quaternion difference between two quaternions.

        Args:
            quat_curr (np.ndarray): Current quaternion in format (x, y, z, w) as default.
            quat_des (np.ndarray): destination quaternion in format (x, y, z, w) as default.
            flip (bool): Whether to flip the order of quaternion. Default: False.
                if the format of q is (w, x, y, z), then flip=True
        Returns:
            np.ndarray: Quaternion difference.
        """
        # Calculate quaternion difference
        r_curr = cls.quat2mat(quat_curr, flip=flip)
        r_des = cls.quat2mat(quat_des, flip=flip)
        r_diff = r_des.inv().dot(r_curr)
        q_diff = r_diff.as_quat()
        return q_diff

    @classmethod
    def euler_diff(cls, euler_curr:Union[np.ndarray, List[float]], 
                  euler_des:Union[np.ndarray, List[float]],
                  order:str='xyz',
                  degrees:bool=False) -> np.ndarray:
        """Calculate the Euler angle difference between two Euler angles.

        Args:
            euler_curr (np.ndarray): Current Euler angles.
            euler_des (np.ndarray): destination Euler angles.
            order (str): Order of Euler angles. Default: 'xyz'.
            degrees (bool): Whether to return the angles in degrees. Default: False.
        Returns:
            np.ndarray: Euler angle difference.
        """
        # Calculate Euler angle difference
        r_curr = cls.euler2mat(euler_curr, order=order, degrees=degrees)
        r_des = cls.euler2mat(euler_des, order=order, degrees=degrees)
        r_diff = r_des.inv().dot(r_curr)
        euler_diff = r_diff.as_euler(order, degrees=degrees)
        return euler_diff
    
    @classmethod
    def mat_diff(cls, mat_curr:Union[np.ndarray, List[float]], 
                mat_des:Union[np.ndarray, List[float]]) -> np.ndarray:
        """Calculate the rotation matrix difference between two rotation matrices.

        Args:
            mat_curr (np.ndarray): Current rotation matrix.
            mat_des (np.ndarray): destination rotation matrix.
        Returns:
            np.ndarray: Rotation matrix difference.
        """
        # Calculate rotation matrix difference
        r_curr = R.from_matrix(mat_curr)
        r_des = R.from_matrix(mat_des)
        r_diff = r_des.inv().dot(r_curr)
        mat_diff = r_diff.as_matrix()
        return mat_diff