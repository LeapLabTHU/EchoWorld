from typing import Union, List
import numpy as np

from .rotation import Rotation


class Transformation:
    """
    Transformation class for calculating transformation matrix, hexagon vectors and heptagon vectors.
    """
    def __init__(self) -> None:
        pass

    @classmethod
    def rotate(cls, 
               theta: float,
               axis: str,
               degrees: bool = True) -> np.ndarray:
        """
        calculate the transformation matrix (4x4) for rotating with a given axis (X, Y, or Z) and angle 
        
        Parameters:
            theta (float): The rotation angle
            axis (str): 'x', 'y', or 'z' axis
            degrees (bool): whether the input rotation angle in degrees, default is true.
    
        Returns:
            np.array: The 4x4 transformation matrix.
        """
        matrix = np.eye(4)
        rotation = Rotation.rotate(theta, axis, degrees)
        matrix[:3, :3] = rotation
        return matrix
    
    @classmethod
    def hexa2trans(cls, hexa: Union[np.ndarray, List[float]],
                   order:str='xyz',
                   degrees:bool=True) -> np.ndarray:
        """
        Convert a hexagon vector to a transformation matrix.
        :param hexa: a vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
                translation: (x, y, z)
                rotation: (x, y, z)
        :param degrees: whether the rotation is in degrees or radians
        :param order: the order of the rotation
        :return: a transformation matrix of shape (4, 4)
        """
        # Extract translation and rotation values
        translation = np.array(hexa[:3])
        rotation = hexa[3:]

        # Convert rotation angles to a rotation matrix
        rotation_matrix = Rotation.euler2mat(rotation, order=order, degrees=degrees)
        
        # Create the transformation matrix
        transformation_matrix = np.eye(4)
    
        # Insert the translation vector
        transformation_matrix[:3, 3] = translation
    
        # Insert the rotation matrix
        transformation_matrix[:3, :3] = rotation_matrix

        return transformation_matrix

    @classmethod
    def trans2hexa(cls, transformation_matrix: Union[np.ndarray, List[List[float]]],
                   order:str='xyz',
                   degrees:bool=True) -> np.ndarray:
        """
        Convert a transformation matrix to a vector.
        :param transformation_matrix: a transformation matrix of shape (4, 4)
        :param degrees: whether the rotation is in degrees or radians
        :param order: the order of the rotation
        :return: a vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
                translation: (x, y, z)
                rotation: (x, y, z)
        """
        # Extract translation and rotation values
        translation = np.array(transformation_matrix[:3, 3])
        rotation_matrix = transformation_matrix[:3, :3]

        # Convert rotation matrix to rotation angles
        rotation = Rotation.mat2euler(rotation_matrix, order=order, degrees=degrees)

        # Create the vector
        vector = np.zeros(6)
    
        # Insert the translation vector
        vector[:3] = translation
    
        # Insert the rotation matrix
        vector[3:] = rotation

        return vector

    @classmethod
    def hepta2trans(cls, hepta:Union[np.ndarray, List[float]],
                   flip:bool=False) -> np.ndarray:
        """
        Convert a heptagon vector to a transformation matrix.
        :param hexa: a vector of length 7, where the first 3 elements are the translation and the last 3 elements are the rotation.
                translation: (x, y, z)
                quaternion: (x, y, z, w)
        :param flip: whether to flip the order of quaternion, default: False
            if the quaternion is (w, x, y, z), then flip=True
        :return: a transformation matrix of shape (4, 4)
        """
        # Extract translation and quaternion values
        translation = np.array(hepta[:3])
        quat = hepta[3:]

        # Convert quaternion to a rotation matrix
        rotation_matrix = Rotation.quat2mat(quat, flip=flip)
        
        # Create the transformation matrix
        transformation_matrix = np.eye(4)
    
        # Insert the translation vector
        transformation_matrix[:3, 3] = translation
    
        # Insert the rotation matrix
        transformation_matrix[:3, :3] = rotation_matrix

        return transformation_matrix

    @classmethod
    def trans2hepta(cls, transformation_matrix: Union[np.ndarray, List[List[float]]],
                   flip:bool=False) -> np.ndarray:
        """
        Convert a transformation matrix to a vector.
        :param transformation_matrix: a transformation matrix of shape (4, 4)
        :param flip: whether to flip the order of quaternion, default: False
            if the quaternion is (w, x, y, z), then flip=True
        :return: a vector of length 7, where the first 3 elements are the translation and the last 3 elements are the rotation.
                translation: (x, y, z)
                quaternion: (x, y, z, w)
        """
        # Extract translation and rotation values
        translation = np.array(transformation_matrix[:3, 3])
        rotation_matrix = transformation_matrix[:3, :3]

        # Convert rotation matrix to quaternion
        quat = Rotation.mat2quat(rotation_matrix, flip=flip)

        # Create the vector
        vector = np.zeros(7)
    
        # Insert the translation vector
        vector[:3] = translation
    
        # Insert the rotation matrix
        vector[3:] = quat

        return vector
    
    @classmethod
    def hexa2hepta(cls, hexa: Union[np.ndarray, List[float]],
                   order:str='xyz',
                   degrees:bool=True,
                   flip:bool=False) -> np.ndarray:
        """
        Convert a hexagon vector to a heptagon vector.
        :param hexa: a vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
                translation: (x, y, z)
                rotation: (x, y, z)
        :param degrees: whether the rotation is in degrees or radians
        :param order: the order of the rotation
        :param flip: whether to flip the order of quaternion, default: False
            if the quaternion is (w, x, y, z), then flip=True
        :return: a vector of length 7, where the first 3 elements are the translation and the last 3 elements are the rotation.
                translation: (x, y, z)
                quaternion: (x, y, z, w)
        """
        trans = cls.hexa2trans(hexa, order, degrees)
        hepta = cls.trans2hepta(trans, flip)
        
        return hepta
    
    @classmethod
    def hepta2hexa(cls, hepta:Union[np.ndarray, List[float]],
                   order:str='xyz',
                   degrees:bool=True,
                   flip:bool=False) -> np.ndarray:
        """
        Convert a heptagon vector to a hexagon vector.
        :param hexa: a vector of length 7, where the first 3 elements are the translation and the last 3 elements are the rotation.
                translation: (x, y, z)
                quaternion: (x, y, z, w)
        :param degrees: whether the rotation is in degrees or radians
        :param order: the order of the rotation
        :param flip: whether to flip the order of quaternion, default: False
            if the quaternion is (w, x, y, z), then flip=True
        :return: a vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
                translation: (x, y, z)
                rotation: (x, y, z)
        """
        trans = cls.hepta2trans(hepta, flip)
        hexa = cls.trans2hexa(trans, order=order, degrees=degrees)

        return hexa

    @classmethod
    def trans_mat(cls, transformation_matrix_1: Union[np.ndarray, List[List[float]]],
                  transformation_matrix_2: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        calculate the transformation matrix of two transformation matrices
        0T2 = 0T1 * 1T2

        :param transformation_matrix_1: the 1st transformation matrix of shape (4, 4)
        :param transformation_matrix_2: the 2nd transformation matrix of shape (4, 4)
        :return: the transformation matrix of shape (4, 4)
        """
        # Convert to numpy array
        if isinstance(transformation_matrix_1, list):
            transformation_matrix_1 = np.array(transformation_matrix_1)
        if isinstance(transformation_matrix_2, list):
            transformation_matrix_2 = np.array(transformation_matrix_2)

        matrix_mat = np.matmul(transformation_matrix_1, transformation_matrix_2)

        return matrix_mat

    @classmethod
    def hexa_mat(cls, hexa_vector_1: Union[np.ndarray, List[float]],
                 hexa_vector_2: Union[np.ndarray, List[float]],
                 order:str='xyz',
                 degrees:bool=True) -> np.ndarray:
        """
        calculate the product of two hexagon vectors
        0V2 = 0V1 * 1V2

        :param hexa_vector_1: the 1st hexagon vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
                    translation: (x, y, z)
                    rotation: (x, y, z)
        :param hexa_vector_2: the 2nd hexagon vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
        :param degrees: whether the rotation is in degrees or radians
        :param order: the order of the rotation
        :return the hexagon vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
        """
        
        # Convert to transformation matrix
        transformation_matrix_1 = cls.hexa2trans(hexa_vector_1, order=order, degrees=degrees)
        transformation_matrix_2 = cls.hexa2trans(hexa_vector_2, order=order, degrees=degrees)

        # Calculate the product of two transformation matrices
        transformation_mat = cls.trans_mat(transformation_matrix_1, transformation_matrix_2)
        
        # Convert to hexagon vector
        hexa_vector = cls.trans2hexa(transformation_mat, order=order, degrees=degrees)

        return hexa_vector

    @classmethod
    def hepta_mat(cls, hepta_vector_1: Union[np.ndarray, List[float]],
                  hepta_vector_2: Union[np.ndarray, List[float]],
                  flip:bool=False) -> np.ndarray:
        """
        calculate the product of two heptagon vectors
        0V2 = 0V1 * 1V2

        :param hepta_vector_1: the 1st heptagon vector of length 7, where the first 3 elements are the translation and the last 4 elements are the quaternion.
                    translation: (x, y, z)
                    quaternion: (x, y, z, w)
        :param hepta_vector_2: the 2nd heptagon vector of length 7, where the first 3 elements are the translation and the last 4 elements are the quaternion.
        :param flip: whether to flip the order of quaternion, default: False
            if the quaternion is (w, x, y, z), then flip=True
        :return the heptagon vector of length 7, where the first 3 elements are the translation and the last 4 elements are the quaternion.
        """
        # convert to transformation matrix
        transformation_matrix_1 = cls.hepta2trans(hepta_vector_1, flip=flip)
        transformation_matrix_2 = cls.hepta2trans(hepta_vector_2, flip=flip)

        # Calculate the product of two transformation matrices
        transformation_mat = cls.trans_mat(transformation_matrix_1, transformation_matrix_2)

        # Convert to heptagon vector
        hepta_vector = cls.trans2hepta(transformation_mat, flip=flip)

        return hepta_vector

    @classmethod
    def trans_diff(cls, transformation_matrix_1: Union[np.ndarray, List[List[float]]],
                  transformation_matrix_2: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        calculate the transformation matrix difference of two transformation matrices
        1T2 = inv(0T1) * 0T2
        
        :param transformation_matrix_1: the 1st transformation matrix of shape (4, 4)
        :param transformation_matrix_2: the 2nd transformation matrix of shape (4, 4)
        :return: the transformation difference matrix of shape (4, 4)
        """
        # Convert to numpy array
        if isinstance(transformation_matrix_1, list):
            transformation_matrix_1 = np.array(transformation_matrix_1)
        if isinstance(transformation_matrix_2, list):
            transformation_matrix_2 = np.array(transformation_matrix_2)
        
        matrix_difference = np.matmul(np.linalg.inv(transformation_matrix_1), transformation_matrix_2)
        
        return matrix_difference
    
    @classmethod
    def trans_diff_inv(cls, transformation_matrix_1: Union[np.ndarray, List[List[float]]],
                  transformation_matrix_2: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        calculate the transformation matrix difference of two transformation matrices
        1T2 = 1T0 * inv(2T0)
        
        :param transformation_matrix_1: the 1st transformation matrix of shape (4, 4)
        :param transformation_matrix_2: the 2nd transformation matrix of shape (4, 4)
        :return: the transformation difference matrix of shape (4, 4)
        """
        # Convert to numpy array
        if isinstance(transformation_matrix_1, list):
            transformation_matrix_1 = np.array(transformation_matrix_1)
        if isinstance(transformation_matrix_2, list):
            transformation_matrix_2 = np.array(transformation_matrix_2)
        
        matrix_difference = np.matmul(transformation_matrix_1, np.linalg.inv(transformation_matrix_2))
        
        return matrix_difference
    
    @classmethod
    def trans_diff_in_hexa(cls, transformation_matrix_1: Union[np.ndarray, List[List[float]]],
                  transformation_matrix_2: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
    
        """
        calculate the transformation matrix difference of two transformation matrices and return the hexagon vector
        1T2 = inv(0T1) * 0T2

        :param transformation_matrix_1: the 1st transformation matrix of shape (4, 4)
        :param transformation_matrix_2: the 2nd transformation matrix of shape (4, 4)
        :return: the hexagon vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
                translation: (x, y, z)
                rotation: (x, y, z)
        """
        matrix_difference = cls.trans_diff(transformation_matrix_1, transformation_matrix_2)
        hexa_diff = cls.trans2hexa(matrix_difference)

        return hexa_diff

    @classmethod
    def hexa_diff(cls, hexa_vector_1: Union[np.ndarray, List[float]],
                  hexa_vector_2: Union[np.ndarray, List[float]],
                  order:str='xyz',
                  degrees:bool=True) -> np.ndarray:
        """
        calculate the difference of two hexagon vectors
        1V2 = inv(0V1) * 0V2

        :param hexa_vector_1: the 1st hexagon vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
                    translation: (x, y, z)
                    rotation: (x, y, z)
        :param hexa_vector_2: the 2nd hexagon vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
        :param order: the order of the rotation
        :param degrees: whether the rotation is in degrees or radians
        :return the hexagon vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
        """
        # Convert to transformation matrix
        transformation_matrix_1 = cls.hexa2trans(hexa_vector_1, order=order, degrees=degrees)
        transformation_matrix_2 = cls.hexa2trans(hexa_vector_2, order=order, degrees=degrees)

        # Calculate the difference of two transformation matrices
        mat_diff = cls.trans_diff(transformation_matrix_1, transformation_matrix_2)
        
        hexa_vector = cls.trans2hexa(mat_diff, order=order, degrees=degrees)

        return hexa_vector
    
    @classmethod
    def hexa_diff_inv(cls, hexa_vector_1: Union[np.ndarray, List[float]],
                      hexa_vector_2: Union[np.ndarray, List[float]],
                      order:str='xyz',
                      degrees:bool=True) -> np.ndarray:
        """
        calculate the difference of two hexagon vectors
        1V2 = 1V0 * inv(2V0)

        :param hexa_vector_1: the 1st hexagon vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
                    translation: (x, y, z)
                    rotation: (x, y, z)
        :param hexa_vector_2: the 2nd hexagon vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
        :param order: the order of the rotation
        :param degrees: whether the rotation is in degrees or radians
        :return the hexagon vector of length 6, where the first 3 elements are the translation and the last 3 elements are the rotation.
        """
        # Convert to transformation matrix
        transformation_matrix_1 = cls.hexa2trans(hexa_vector_1, order=order, degrees=degrees)
        transformation_matrix_2 = cls.hexa2trans(hexa_vector_2, order=order, degrees=degrees)

        # Calculate the difference of two transformation matrices
        mat_diff = cls.trans_diff_inv(transformation_matrix_1, transformation_matrix_2)
        
        hexa_vector = cls.trans2hexa(mat_diff, order=order, degrees=degrees)

        return hexa_vector

    @classmethod
    def hepta_diff(cls, hepta_vector_1: Union[np.ndarray, List[float]],
                   hepta_vector_2: Union[np.ndarray, List[float]],
                   flip:bool=False) -> np.ndarray:
        """
        calculate the difference of two heptagon vectors
        1V2 = inv(0V1) * 0V2

        :param hepta_vector_1: the 1st heptagon vector of length 7, where the first 3 elements are the translation and the last 4 elements are the quaternion.
                    translation: (x, y, z)
                    quaternion: (x, y, z, w)
        :param hepta_vector_2: the 2nd heptagon vector of length 7, where the first 3 elements are the translation and the last 4 elements are the quaternion.
        :param flip: whether to flip the order of quaternion, default: False
            if the quaternion is (w, x, y, z), then flip=True
        :return the heptagon vector of length 7, where the first 3 elements are the translation and the last 4 elements are the quaternion.
        """
        # Convert to transformation matrix
        transformation_matrix_1 = cls.hepta2trans(hepta_vector_1, flip=flip)
        transformation_matrix_2 = cls.hepta2trans(hepta_vector_2, flip=flip)

        # Calculate the difference of two transformation matrices
        mat_diff = cls.trans_diff(transformation_matrix_1, transformation_matrix_2)

        hepta_vector = cls.trans2hepta(mat_diff, flip=flip)

        return hepta_vector
    
    @classmethod
    def hepta_diff_inv(cls, hepta_vector_1: Union[np.ndarray, List[float]],
                   hepta_vector_2: Union[np.ndarray, List[float]],
                   flip:bool=False) -> np.ndarray:
        """
        calculate the difference of two heptagon vectors
        1V2 = 1V0 * inv(2V0)

        :param hepta_vector_1: the 1st heptagon vector of length 7, where the first 3 elements are the translation and the last 4 elements are the quaternion.
                    translation: (x, y, z)
                    quaternion: (x, y, z, w)
        :param hepta_vector_2: the 2nd heptagon vector of length 7, where the first 3 elements are the translation and the last 4 elements are the quaternion.
        :param flip: whether to flip the order of quaternion, default: False
            if the quaternion is (w, x, y, z), then flip=True
        :return the heptagon vector of length 7, where the first 3 elements are the translation and the last 4 elements are the quaternion.
        """
        # Convert to transformation matrix
        transformation_matrix_1 = cls.hepta2trans(hepta_vector_1, flip=flip)
        transformation_matrix_2 = cls.hepta2trans(hepta_vector_2, flip=flip)

        # Calculate the difference of two transformation matrices
        mat_diff = cls.trans_diff_inv(transformation_matrix_1, transformation_matrix_2)

        hepta_vector = cls.trans2hepta(mat_diff, flip=flip)

        return hepta_vector