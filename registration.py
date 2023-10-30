import trimesh
import trimesh.registration
import trimesh.transformations
import os.path

import numpy as np

from abc import ABC, abstractmethod
from scipy.linalg import orthogonal_procrustes
from scipy import sparse

import utils


class Registerer(ABC):
    """ Abstract class for registration algorithms."""
    def __init__(self, reference_path, reference_landmarks_path=None,
                 show_results=False):
        self._reference_mesh = utils.load_trimesh(reference_path)
        if reference_landmarks_path is not None:
            self._reference_landmarks = utils.load_landmarks(
                reference_landmarks_path, mesh=self._reference_mesh)
        else:
            self._reference_landmarks = None
        self._show_results = show_results

    def __call__(self, mesh_path, mesh_landmarks_path=None, 
                 out_path=None, **kwargs):
        mesh = utils.load_trimesh(mesh_path)
        if mesh_landmarks_path is not None:
            lms = utils.load_landmarks(mesh_landmarks_path, mesh=mesh)
        else:
            lms = None
        registered_mesh = self.register(mesh, lms)
        if out_path is not None:
            registered_mesh.export(out_path)
        return registered_mesh

    @abstractmethod
    def register(self, mesh, mesh_landmarks=None, **kwargs) -> trimesh.Trimesh:
        pass

    @property
    def reference_mesh(self):
        return self._reference_mesh

    @property
    def reference_landmarks(self):
        return self._reference_landmarks

    def register_all_and_save(self, meshes_dir, landmarks_dir=None, **kwargs):
        self._show_results = False
        out_dir = os.path.join(meshes_dir, "registered")

        all_files = utils.find_filenames(meshes_dir)
        for fname in all_files:
            if landmarks_dir is not None:
                # it assumes that mesh and landmarks have the same name,
                # but different formats
                lms_path = os.path.join(landmarks_dir,
                                        os.path.split(fname)[1][:-4] + ".txt")
            else:
                # it assumes that mesh and landmarks are saved in the same
                # folder, with the same name and different formats
                lms_path = fname[:-4] + ".txt"
            if not os.path.isfile(lms_path):
                lms_path = None

            registered_mesh = self.__call__(fname, lms_path, **kwargs)

            registered_mesh.export(
                os.path.join(out_dir, os.path.split(fname)[1]))

    def show_results(self, registered_mesh, comparison_mesh=None):
        scene = trimesh.Scene()
        if comparison_mesh is None:
            scene.add_geometry(self._reference_mesh)
            if self._reference_landmarks is not None:
                scene.add_geometry(
                    trimesh.points.PointCloud(self._reference_landmarks))
        else:
            scene.add_geometry(comparison_mesh)
        scene.add_geometry(registered_mesh)
        scene.show()


class ProcrustesLandmarkRegisterer(Registerer):
    """ Class performing Procrustes registration of two meshes.
    The transformation is computed between their landmarks."""
    def register(self, mesh, mesh_landmarks=None, return_landmarks=False):
        assert mesh_landmarks is not None
        assert self._reference_landmarks is not None
        lms = mesh_landmarks

        translation_ref = np.mean(self._reference_landmarks, 0)
        centered_ref_lms = self._reference_landmarks - translation_ref
        norm_ref_lms = np.linalg.norm(centered_ref_lms)
        centered_ref_lms /= norm_ref_lms

        translation = np.mean(lms, 0)
        centered_lms = lms - translation
        norm_lms = np.linalg.norm(centered_lms)
        centered_lms /= norm_lms

        rotation, scale = orthogonal_procrustes(centered_ref_lms, centered_lms)

        reg_m_verts = mesh.vertices - translation
        reg_m_verts /= norm_lms
        reg_m_verts = np.dot(reg_m_verts, rotation.T) * scale
        reg_m_verts = (reg_m_verts * norm_ref_lms) + translation_ref

        registered_mesh = mesh.copy()
        registered_mesh.vertices = reg_m_verts

        if self._show_results:
            self.show_results(registered_mesh)

        if return_landmarks:
            reg_m_lms = np.dot(centered_lms, rotation.T) * scale
            reg_m_lms = (reg_m_lms * norm_ref_lms) + translation_ref
            return registered_mesh, reg_m_lms
        else:
            return registered_mesh


class InertiaAxesAndIcpRegisterer(Registerer):
    """ Class that wraps trimesh.registration.mesh_other().
    Align a mesh with another mesh or a PointCloud using
    the principal axes of inertia as a starting point which
    is refined by iterative closest point.
    """
    def register(self, mesh, mesh_landmarks=None,
                 samples=500, scale=True,
                 icp_first=10, icp_final=50) -> trimesh.Trimesh:

        matrix, _ = trimesh.registration.mesh_other(mesh, self._reference_mesh,
                                                    samples, scale,
                                                    icp_first, icp_final)

        registered_mesh = mesh.copy()
        registered_mesh.vertices = \
            trimesh.transformations.transform_points(mesh.vertices, matrix)

        if self._show_results:
            self.show_results(registered_mesh)

        return registered_mesh


class ProcrustesLandmarkAndIcpRegisterer(Registerer):
    """ Class to perform ICP after an initial landmark-based Procrustes
    registration. ICP is wrapping the trimesh implementation:
    trimesh.registration.icp().
    """
    def __init__(self, reference_path, reference_landmarks_path,
                 show_results=False):
        super().__init__(reference_path, reference_landmarks_path, show_results)
        self._p_registerer = ProcrustesLandmarkRegisterer(
            reference_path, reference_landmarks_path)

    def register(self, mesh, mesh_landmarks=None,
                 samples=500, threshold=1e-5,
                 max_iterations=20) -> trimesh.Trimesh:

        landmark_aligned = self._p_registerer.register(mesh, mesh_landmarks)

        if samples > 0:
            vertices = landmark_aligned.sample(samples)
        else:
            vertices = landmark_aligned.vertices

        matrix, _, _ = trimesh.registration.icp(
            vertices, self._reference_mesh,
            threshold=threshold, max_iterations=max_iterations
        )

        registered_mesh = landmark_aligned.copy()
        registered_mesh.vertices = trimesh.transformations.transform_points(
            landmark_aligned.vertices, matrix)

        if self._show_results:
            self.show_results(registered_mesh)

        return registered_mesh


class ProcrustesLandmarkAndNicpRegisterer(Registerer):
    """ Class to perform Non-rigid ICP after an initial landmark-based
    Procrustes registration. The NICP implementation is inspired by menpo3d:
    https://github.com/menpo/menpo3d/blob/6650918e786ac98112387b97f5ecf8cc67025
        f9f/menpo3d/correspond/nicp.py#L243
    Our implementation is not bounded to the menpo library, and it does
    not require the landmarks to follow the ibug conventions
    (although advised for heads and faces).
    The new mesh is first aligned to the reference, then the reference is non
    rigidly deformed to match the new mesh
    """
    def __init__(self, reference_path, reference_landmarks_path,
                 show_results=False):
        super().__init__(reference_path, reference_landmarks_path, show_results)
        self._p_registerer = ProcrustesLandmarkRegisterer(
            reference_path, reference_landmarks_path)

    def register(self, mesh, mesh_landmarks=None, eps=1e-3, max_iters=8,
                 stiffness_weights=None, data_weights=None,
                 landmark_weights=None) -> trimesh.Trimesh:
        """
        Calling this function the reference mesh is fitted onto 'mesh'
        :param mesh: Trimesh to fit
        :param mesh_landmarks: landmarks of the mesh to fit
        :param eps: float. If the error decrease is inferior to this value,
            the current step ends.
        :param max_iters: maximum number of iterations per step.
        :param stiffness_weights: these weights can be provided either as a
            list of scalars that equally weight all edges at each step, or
            as per-vertex values, thus enabling more control over regional
            deformations. The length of the list determines the number of steps
            of the algorithm (i.e. how many times the algorithm runs with a
            specific set of weights). It should have the same length as
            data_weights and landmark_weights. If None, default values
            are used.
        :param data_weights: these weights can be provided either as a list of
            scalars that equally weight all vertex normals at each step, or
            as per-vertex values, thus enabling more control over regional
            deformations. The length of the list determines the number of steps
            of the algorithm (i.e. how many times the algorithm runs with a
            specific set of weights). It should have the same length as
            stiffness_weights and landmark_weights. If None, default values
            are used.
        :param landmark_weights: list of scala weights to use at every step of
            the algorithm to control the influence of the landmarks over
            the registration. The length of the list determines the number of
            steps of the algorithm (i.e. how many times the algorithm runs with
            a specific set of weights). It should have the same length as
            stiffness_weights and data_weights. If None, default values
            are used.

        :return: Trimesh obtained non-rigidly registering vertices of the
            reference mesh onto the mesh geometry. Ideally, this method should
            return a mesh with the same geometry of the 'mesh' and the same
            topology of the 'reference' mesh.
        """
        landmark_aligned, mesh_landmarks = self._p_registerer.register(
            mesh, mesh_landmarks, return_landmarks=True)

        reference_verts = self._reference_mesh.vertices
        reference_trilist = self._reference_mesh.faces
        mesh_verts = landmark_aligned.vertices
        mesh_normals = mesh.vertex_normals

        # Scale meshes and their landmarks #####################################
        # Scale factors completely change the behavior of the algorithm
        # rescale the reference down to a sensible size
        # (so it fits inside box of diagonal 1) and is centred on the origin.
        # This is undone after the fit.
        tr = np.mean(reference_verts, axis=0)
        reference_bounds_diff = \
            np.max(reference_verts, axis=0) - np.min(reference_verts, axis=0)
        sc = np.sqrt(np.sum(reference_bounds_diff ** 2))

        reference_verts = (reference_verts - tr) / sc
        mesh_verts = (mesh_verts - tr) / sc
        mesh_landmarks = (mesh_landmarks - tr) / sc
        reference_landmarks = (self._reference_landmarks - tr) / sc
        ########################################################################

        # Prepare weights ######################################################
        if stiffness_weights is None:
            stiffness_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]

        n_iterations = len(stiffness_weights)

        if mesh_landmarks is not None and landmark_weights is None:
            landmark_weights = [5, 2, 0.5, 0, 0, 0, 0, 0]
        elif landmark_weights is None:
            landmark_weights = [None] * n_iterations

        if data_weights is None:
            data_weights = [None] * n_iterations
        ########################################################################

        # Prepare all info that can be computed before looping #################

        n_dims = reference_verts.shape[1]
        h_dims = n_dims + 1  # Homogeneous dimension
        n = reference_verts.shape[0]

        mat_s, unique_edge_pairs = self.reference_node_arc_incidence_matrix()

        # weight matrix
        weight_mat = np.identity(n_dims + 1)
        weight_mat_kron_s = sparse.kron(mat_s, weight_mat)

        # init transformation
        x_prev = np.tile(np.zeros((n_dims, h_dims)), n).T
        current_verts = reference_verts

        # prepare some indices for efficient construction of the sparse matrices
        row = np.hstack(
            (np.repeat(np.arange(n)[:, None], n_dims, axis=1).ravel(),
             np.arange(n))
        )
        x = np.arange(n * h_dims).reshape((n, h_dims))
        col = np.hstack((x[:, :n_dims].ravel(), x[:, n_dims]))
        o = np.ones(n)

        if mesh_landmarks is not None:
            reference_lm_index = utils.closest_indices_to_landmarks(
                reference_verts, reference_landmarks)
            mesh_lms = mesh_landmarks
            n_landmarks = mesh_lms.shape[0]
            lm_mask = np.in1d(row, reference_lm_index)
            col_lm = col[lm_mask]
            row_lm_to_fix = row[lm_mask]
            reference_lm_index_l = list(reference_lm_index)
            row_lm = np.array(
                [reference_lm_index_l.index(r) for r in row_lm_to_fix])
        else:
            mesh_lms, n_landmarks, lm_mask, = None, None, None
            row_lm, col_lm = None, None
        ########################################################################

        for alpha, beta, gamma in zip(
                stiffness_weights, landmark_weights, data_weights):

            alpha_is_per_vertex = isinstance(alpha, np.ndarray)
            if alpha_is_per_vertex:  # stiffness is provided per-vertex
                if alpha.shape[0] != n:
                    raise ValueError()
                alpha_per_edge = alpha[unique_edge_pairs].mean(axis=1)
                alpha_mat_s = sparse.diags(alpha_per_edge).dot(mat_s)
                alpha_weight_mat_kron_s = sparse.kron(alpha_mat_s, weight_mat)
            else:  # stiffness is global
                alpha_weight_mat_kron_s = alpha * weight_mat_kron_s

            j = 0
            while True:  # iterate until convergence
                j += 1  # track the iterations for this alpha/landmark weight

                nearest_verts, nearest_indices = \
                    utils.find_nearest_verts_and_indices(
                        current_verts, mesh_verts)
                nearest_normals = mesh_normals[nearest_indices]

                # Calculate the normals of the current current_verts
                current_mesh = trimesh.Trimesh(current_verts,
                                               faces=reference_trilist)
                current_normals = current_mesh.vertex_normals

                # If the dot of the normals < 0.9 don't contrib to deformation
                normals_weight_mat = \
                    (nearest_normals * current_normals).sum(axis=1) > 0.9

                if gamma is not None:
                    normals_weight_mat = normals_weight_mat * gamma

                # Build the sparse diagonal weight matrix
                normals_weight_mat_s = sparse.diags(
                    normals_weight_mat.astype(float)[None, :], [0])

                data = np.hstack((current_verts.ravel(), o))
                data_s = sparse.coo_matrix((data, (row, col)))

                to_stack_a = [alpha_weight_mat_kron_s,
                              normals_weight_mat_s.dot(data_s)]
                to_stack_b = [np.zeros((alpha_weight_mat_kron_s.shape[0],
                                        n_dims)),
                              nearest_verts * normals_weight_mat[:, None]]

                if mesh_landmarks is not None:
                    lms_data_mat_s = sparse.coo_matrix(
                        (data[lm_mask], (row_lm, col_lm)),
                        shape=(n_landmarks, data_s.shape[1])
                    )
                    to_stack_a.append(beta * lms_data_mat_s)
                    to_stack_b.append(beta * mesh_lms)

                a_s = sparse.vstack(to_stack_a).tocsr()
                b_s = sparse.vstack(to_stack_b).tocsr()

                x = utils.sparse_solve(a_s, b_s)

                # deform template
                current_verts = data_s.dot(x)

                err = np.linalg.norm(x_prev - x, ord="fro")
                stop_criterion = err / np.sqrt(np.size(x_prev))

                x_prev = x

                if stop_criterion < eps or j > max_iters:
                    break

        registered_mesh = self._reference_mesh.copy()
        registered_mesh.vertices = current_verts * sc + tr

        if self._show_results:
            self.show_results(registered_mesh, comparison_mesh=landmark_aligned)
            registered_mesh.show()

        return registered_mesh

    def reference_node_arc_incidence_matrix(self):
        unique_edge_pairs = self._reference_mesh.edges_unique
        m = unique_edge_pairs.shape[0]

        # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
        row = np.hstack((np.arange(m), np.arange(m)))
        col = unique_edge_pairs.T.ravel()
        data = np.hstack((-1 * np.ones(m), np.ones(m)))
        return sparse.coo_matrix((data, (row, col))), unique_edge_pairs


if __name__ == "__main__":
    rmp = "/home/simo/Desktop/bws_project/template/template_sym2_fixed.ply"
    rlp = "/home/simo/Desktop/bws_project/template/template_sym2_fixed_ibug_68.txt"

    mp = "/media/simo/DATASHURPRO/old_unused_pre_post/SD-VAE Frontofacial Outcomes/Pilot Data/Meshes/1/626491 Post-Op STL.stl"
    lp = "/media/simo/DATASHURPRO/old_unused_pre_post/SD-VAE Frontofacial Outcomes/Pilot Data/Landmarks/1/626491 Post-Op Landmarks.txt"

    # registerer = ProcrustesLandmarkRegisterer(reference_path=rmp,
    #                                           reference_landmarks_path=rlp)
    # registerer = InertiaAxesAndIcpRegisterer(reference_path=rmp)
    # registerer = ProcrustesLandmarkAndIcpRegisterer(
    #     reference_path=rmp, reference_landmarks_path=rlp, show_results=True)
    registerer = ProcrustesLandmarkAndNicpRegisterer(
        reference_path=rmp, reference_landmarks_path=rlp, show_results=True)
    registerer(mp, lp)
    print("done")



