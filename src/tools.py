import os
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R

import yaml
import matplotlib.cm as cm

GRAVITY_CONSTANT = 9.80665
SPEED_KM_TO_MS = 1000/3600

def check_file_path(*argv)->str:

    path = os.path.join(*argv)
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError(f"File path '{path}' does not exist.")

def reject_outliers(data, m = 5.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

    

def run_command(command:str)->None:

    print("Executing command: ", command)

    process = subprocess.run(command, shell=True, capture_output=True, text=True)

    print(process.stdout)

    if process.returncode != 0:

        print(f"Error executing command: {process.stderr}")

    else:

        print("Command executed successfully.")



def apply_transform(H, points):

   

    # apply the transformation to the points

    points = np.array(points)

    points = np.hstack([points, np.ones((points.shape[0], 1))])

    points = np.dot(H, points.T).T

    return points[:, :3]



def load_calib_file(path):

    """load path on this format %YAML:1.0

---

camera_name: "1709940771.408448"

image_width: 640

image_height: 480

camera_matrix:

   rows: 3

   cols: 3

   data: [ 3.7769427490234375e+02, 0., 3.2287564086914062e+02, 0., 0.,

       3.7732968139648438e+02, 2.4526431274414062e+02, 0., 0. ]

distortion_coefficients:

   rows: 1

   cols: 5

   data: [ 0., 0., 0., 0., 0. ]

distortion_model: plumb_bob

rectification_matrix:

   rows: 3

   cols: 3

   data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]

projection_matrix:

   rows: 3

   cols: 4

   data: [ 3.7769427490234375e+02, 0., 3.2287564086914062e+02, 0., 0.,

       3.7732968139648438e+02, 2.4526431274414062e+02, 0., 0., 0., 1.,

       0. ]

local_transform:

   rows: 3

   cols: 4

   data: [ 3.86178913e-03, 7.16005638e-03, 9.99966919e-01,

       -8.22682559e-05, -9.99987423e-01, 3.22349067e-03, 3.83878709e-03,

       -5.90039231e-02, -3.19589814e-03, -9.99969184e-01, 7.17241457e-03,

       5.36633779e-05 ]

"""

    # print("loading calib file", path)

    check_file_path(path)

    with open(path, 'r') as f:

        # remove the two first line

        f.readline()

        f.readline()

        data = yaml.load(f, Loader=yaml.SafeLoader)

        data["camera_matrix"]["data"] = np.array(data["camera_matrix"]["data"]).reshape((3, 3))

        data["distortion_coefficients"]["data"] = np.array(data["distortion_coefficients"]["data"]).reshape((1, 5))

        data["rectification_matrix"]["data"] = np.array(data["rectification_matrix"]["data"]).reshape((3, 3))

        data["projection_matrix"]["data"] = np.array(data["projection_matrix"]["data"]).reshape((3, 4))

        data["local_transform"]["data"] = np.array(data["local_transform"]["data"]).reshape((3, 4))

    return data



def clear_folder(folder):

    if os.path.exists(folder):

        for file in os.listdir(folder):

            os.remove(os.path.join(folder, file))

    else:

        os.makedirs(folder)



def filename_to_timestamp(string):

    """Convert a filename to a timestamp"""

    timestamp_str = string.split(".")

    if len(timestamp_str) != 3:

        raise ValueError(f"Invalid filename format: {string}")

    timestamp_str = timestamp_str[0] + "." + timestamp_str[1]

    return str(timestamp_str)



def transform_matrix_translation(x,y,z):

    return np.array([[1, 0, 0, x],

                     [0, 1, 0, y],

                     [0, 0, 1, z],

                     [0, 0, 0, 1]])

def transform_matrix_rotation(x,y,z):

    rotation_matrix = R.from_euler('xyz', [x, y, z], degrees=True).as_matrix()

    return np.vstack([np.hstack([rotation_matrix, np.zeros((3, 1))]), np.array([0, 0, 0, 1])])



def encode_float(value,use_color_map):

    if value < 0 or value > 1:

        raise ValueError("Input value must be between 0 and 1.")

    if use_color_map:

        red, green, blue, _ = cm.get_cmap('nipy_spectral')(value)

        return np.array([red, green, blue])

    else:

        return np.array([value, value, value])

    

def decode_float(red, green, blue):

    encoded_value = (red << 16) + (green << 8) + blue

    value = encoded_value / 16**6

    return value

