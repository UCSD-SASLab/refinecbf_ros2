from ament_index_python.packages import get_package_share_directory
import os
import yaml
from typing import Union


def load_parameters(robot, exp, param_type):
    package_dir = get_package_share_directory('refinecbf_ros2')
    if isinstance(exp, int):
            exp = 'exp' + str(exp)
    with open(os.path.join(package_dir, 'config', robot, exp, param_type + '.yaml'), 'r') as file:
        return yaml.safe_load(file)
        
def load_array(robot: str, exp: Union[str, int], array_type: str):
    import numpy as np
    package_dir = get_package_share_directory('refinecbf_ros2')
    if isinstance(exp, int):
        exp = 'exp' + str(exp)
    return np.load(os.path.join(package_dir, 'config', robot, exp, array_type + '.npy'))
