import numpy as np
from typing import Tuple,List

class Pixel_to_cm:
    def __init__(self):
        self.transform_matrix = None

    def fit(self, 
                 origin_pixel: Tuple[float], 
                 ref_points_pixel:List[Tuple[float]],
                 ref_points_cm:List[Tuple[float]]):

        self.dim = len(origin_pixel)
        # create a matrix [point1.T point2.T point3.T]
        ref_matrix_pixel = np.array(ref_points_pixel).T  
        # create a matrix [point1.T point2.T point3.T]
        ref_matrix_cm = np.array(ref_points_cm).T
        # create origin point as column vector (n, 1)
        self.origin_pixel = np.array(origin_pixel)

        assert ref_matrix_cm.shape == ref_matrix_pixel.shape == (self.dim, self.dim),\
            f"The number of reference points {len(ref_matrix_pixel)=}  {len(ref_matrix_pixel)=} should match the number of dimensions {self.dim}."
        try:
            self.transform_matrix = ref_matrix_cm @ np.linalg.inv(ref_matrix_pixel - self.origin_pixel.reshape(-1,1))
        except np.linalg.LinAlgError as e:
            print(
                'The vectors connecting reference points and origin point should be linear independent.'
            )
            raise e
        
        return self

    def transform(self, point_pixel: Tuple[float]) -> Tuple[float]:
        assert self.transform_matrix is not None, 'You need to fit the transformer first.'
        assert len(point_pixel) == self.dim, 'The input point should have the same dimensions.'

        return tuple(self.transform_matrix @ (np.array(point_pixel) - self.origin_pixel))

    def inverse_transform(self, point_cm: Tuple[float]) -> Tuple[int]:
        assert self.transform_matrix is not None, 'You need to fit the transformer first.'
        assert len(point_cm) == self.dim, 'The input point should have the same dimensions.'
        temp = np.round(np.linalg.inv(self.transform_matrix) @ point_cm + self.origin_pixel\
                        ).astype(int)
        return tuple(temp)
        


if __name__=='__main__':
    origin_pixel = (235, 34)
    ref_p1_pixel, ref_p2_pixel = (141, 545), (789, 29)
    ref_p1_cm, ref_p2_cm = (0, 40), (40 , 0)

    transformer = Pixel_to_cm().fit(origin_pixel, [ref_p1_pixel, ref_p2_pixel], [ref_p1_cm, ref_p2_cm])
    
    print(transformer.transform((545, 789)))
    print(transformer.inverse_transform((0, 0)))
    print(transformer.transform((235, 34)))