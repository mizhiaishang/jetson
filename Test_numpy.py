import numpy as np


def test_numpy_distance():
    occupied_positions = np.array([[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0],
                                   [10.0, 11.0, 12.0],
                                   [13.0, 14.0, 15.0]])
    new_position_close = np.array([1.2, 2.1, 3.1])
    new_position_far = np.array([10.0, 11.0, 12.0])
    distance_check = 0.5

    # 用矩阵运算的方法计算距离
    print(occupied_positions.shape, new_position_close.shape)
    distances = np.linalg.norm(occupied_positions - new_position_close, axis=1)
    is_occupied_close = np.any(distances < distance_check)
    print(f"Distances to close position: {distances}, is occupied: {is_occupied_close}")
    assert is_occupied_close, "The close position should be considered occupied."

def main():
    test_numpy_distance()

if __name__ == "__main__":
    main()