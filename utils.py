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


def import_wrap_point_on_triangle_landmarks_as_b_coords(landmarks_path):
    """
    This function imports landmarks which were manually defined in Wrap3D and
    exported as 'point on triangle'. This data format represents each landmark
    as a point on a triangular face of the mesh. Therefore, it stores the index
    of the face on which the landmark belongs, and the barycentric coordinates
    of the landmark with respect to the vertices of the face.

    :param landmarks_path: path to the txt file storing the landmarks
    :return: landmarks as a Nx3 numpy array, where N is the number of landmarks.
    """
    with open(landmarks_path, "r") as f:
        raw = ''.join(f.readlines())
    bar_coords = raw.replace(' ', '').replace('\n', '').strip('[]').split('],[')

    tri_idxs, b_lms = [], []
    for bc in bar_coords:
        bcs = bc.split(',')
        tri_idxs.append(int(bcs[0]))
        b_lms.append([float(bcs[1]), float(bcs[1]),
                      1 - float(bcs[1]) - float(bcs[2])])

    return tri_idxs, np.asarray(b_lms)


def boundary_tri_index(tri_faces):
    """
    Boolean index into triangles that are at the edge of the TriMesh.
    adapted from Menpo shape.

    :param tri_faces: list of triangular faces
    :return: boolean mask saying for each triangle whether any of its edges
        is not also an edge of another triangle (and so this triangle exists on
        the boundary of the mesh)
    """
    # Compute the edge indices so that we can find duplicated edges
    edge_indices = np.hstack((tri_faces[:, [0, 1]],
                              tri_faces[:, [1, 2]],
                              tri_faces[:, [2, 0]])).reshape(-1, 2)
    # Compute the triangle indices and repeat them so that when we loop
    # over the edges we get the correct triangle index per edge
    tri_indices = np.arange(tri_faces.shape[0]).repeat(3)

    # Loop over the edges to find the "lonely" triangles that have an edge
    # that isn't shared with another triangle.
    lonely_triangles = {}
    for edge, t_i in zip(edge_indices, tri_indices):
        # Sorted the edge indices since we may see an edge (0, 1) and then
        # see it again as (1, 0) when in fact that is the same edge
        sorted_edge = tuple(sorted(edge))
        if sorted_edge not in lonely_triangles:
            lonely_triangles[sorted_edge] = t_i
        else:
            # If we've already seen the edge then we will never see it again,
            # so we can just remove it from the candidate set
            del lonely_triangles[sorted_edge]

    mask = np.zeros(tri_faces.shape[0], dtype=bool)
    mask[np.array(list(lonely_triangles.values()))] = True
    return mask


def closest_indices_to_landmarks(vertices, landmarks):
    """
    For each landmark find the index of the closest vertex

    :param vertices: [N, 3] numpy array with vertex coordinates
    :param landmarks: [L, 3] numpy array with landmark coordinates
    :return: list of indices with L elements
    """
    indices = []
    for lnd in landmarks:
        indices.append(np.argmin(np.sum((vertices - lnd) ** 2, axis=1)))
    return indices


# Functions with multiple implementations. The fastest implementation is
# always prioritised. If the library needed is not available use slower option.

try:
    try:
        import torch
        import pykeops.torch

        def find_nearest_verts_and_indices(current_verts, mesh_verts):
            device = torch.device('cuda') if torch.cuda.is_available() else \
                torch.device('cpu')
            x_i = pykeops.torch.LazyTensor(torch.Tensor(
                current_verts[:, None, :]).to(device).contiguous())
            y_j = pykeops.torch.LazyTensor(torch.Tensor(
                mesh_verts[None, :, :]).to(device).contiguous())
            pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
            nearest_indices = pairwise_distance_ij.argKmin(
                K=1, axis=1)[:, 0].cpu().numpy()
            nearest_verts = mesh_verts[nearest_indices, :]
            return nearest_verts, nearest_indices

    except ModuleNotFoundError:
        print("Either torch or keops are not available, trying to use keops "
              "with numpy")
        import pykeops.numpy

        def find_nearest_verts_and_indices(current_verts, mesh_verts):
            x_i = pykeops.numpy.LazyTensor(current_verts[:, None, :])
            y_j = pykeops.numpy.LazyTensor(mesh_verts[None, :, :])
            pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
            nearest_indices = pairwise_distance_ij.argKmin(K=1, axis=1)[:, 0]
            nearest_verts = mesh_verts[nearest_indices, :]
            return nearest_verts, nearest_indices

except ModuleNotFoundError:
    print("keops implementation of K-NN not available as keops has not been"
          "installed. Using pytorch3d implementation instead.")
    import pytorch3d.ops
    import torch

    def find_nearest_verts_and_indices(current_verts, mesh_verts):
        device = torch.device('cuda') if torch.cuda.is_available() else \
            torch.device('cpu')
        _, nearest_indices, nearest_verts = pytorch3d.ops.knn_points(
            torch.Tensor(current_verts).unsqueeze(0).to(device),
            torch.Tensor(mesh_verts).unsqueeze(0).to(device),
            K=1, return_nn=True)
        nearest_indices = nearest_indices[0, :, 0].cpu().numpy()
        nearest_verts = nearest_verts.squeeze(0).cpu().numpy()
        return nearest_verts[:, 0, :], nearest_indices


try:
    from sksparse.cholmod import cholesky_AAt

    def sparse_solve(a_sparse, b_sparse):
        return cholesky_AAt(a_sparse.T)(a_sparse.T.dot(b_sparse)).toarray()

except ModuleNotFoundError:
    print("couldn't find skparse library. Solving sparse system with scipy "
          "sparse solver. This implementation is considerably slower!")
    from scipy.sparse.linalg import spsolve

    def sparse_solve(a_sparse, b_sparse):
        solved = spsolve(a_sparse.T.dot(a_sparse), a_sparse.T.dot(b_sparse))
        return solved.toarray()