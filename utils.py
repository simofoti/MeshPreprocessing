import os
import trimesh
import numpy as np


def find_filenames(root):
    """
    This function finds all meshes withing a folder. Only stl, obj, and ply
    meshes are recognised.

    :param root: root path of the folder
    :return: list of all filenames with their absolute path
    """
    root_l = len(root)
    files = []
    for dirpath, _, fnames in os.walk(root):
        for f in fnames:
            if f.endswith('.ply') or f.endswith('.obj') or f.endswith('.stl'):
                absolute_path = os.path.join(dirpath, f)
                f = absolute_path[dirpath.index(root) + root_l + 1:]
                files.append(f)
    return files


def load_trimesh(mesh_path):
    return trimesh.load_mesh(mesh_path, process=False)


def load_landmarks(landmarks_path, lms_type="point_on_triangle",
                   mesh=None, mesh_path=None):
    if lms_type == "point_on_triangle":
        lms = import_wrap_point_on_triangle_landmarks(landmarks_path, mesh,
                                                      mesh_path)
    else:
        raise NotImplementedError
    return lms


def import_wrap_point_on_triangle_landmarks(landmarks_path,
                                            mesh=None, mesh_path=None):
    """
    This function imports landmarks which were manually defined in Wrap3D and
    exported as 'point on triangle'. This data format represents each landmark
    as a point on a triangular face of the mesh. Therefore, it stores the index
    of the face on which the landmark belongs, and the barycentric coordinates
    of the landmark with respect to the vertices of the face.

    :param landmarks_path: path to the txt file storing the landmarks
    :param mesh: Trimesh mesh on which the landmarks were defined. This
        parameter is optional as mesh_path can be used instead.
    :param mesh_path: path to the mesh on which the landmarks were defined.
    :return: landmarks as a Nx3 numpy array, where N is the number of landmarks.
    """
    with open(landmarks_path, "r") as f:
        raw = ''.join(f.readlines())
    bar_coords = raw.replace(' ', '').replace('\n', '').strip('[]').split('],[')

    if mesh is None:
        assert mesh_path is not None
        mesh = load_trimesh(mesh_path)

    lms = []
    for bc in bar_coords:
        bcs = bc.split(',')
        tri = mesh.faces[int(bcs[0].split('.')[0])]
        lms.append(float(bcs[1]) * mesh.vertices[tri[0]] +
                   float(bcs[2]) * mesh.vertices[tri[1]] +
                   (1 - float(bcs[1]) - float(bcs[2])) * mesh.vertices[tri[2]])

    return np.asarray(lms)

