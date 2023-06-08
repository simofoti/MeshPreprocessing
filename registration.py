import trimesh
import trimesh.registration
import trimesh.transformations
import os.path

import numpy as np

from abc import ABC, abstractmethod
from scipy.linalg import orthogonal_procrustes

import utils


class Registerer(ABC):
    def __init__(self, reference_path, reference_landmarks_path=None,
                 show_results=False):
        self._reference_mesh = utils.load_trimesh(reference_path)
        if reference_landmarks_path is not None:
            self._reference_landmarks = utils.load_landmarks(
                reference_landmarks_path, mesh=self._reference_mesh)
        else:
            self._reference_landmarks = None
        self._show_results = show_results

    def __call__(self, mesh_path, mesh_landmarks_path=None, **kwargs):
        mesh = utils.load_trimesh(mesh_path)
        if mesh_landmarks_path is not None:
            lms = utils.load_landmarks(mesh_landmarks_path, mesh=mesh)
        else:
            lms = None
        return self.register(mesh, lms)

    @abstractmethod
    def register(self, mesh, mesh_landmarks=None, **kwargs) -> trimesh.Trimesh:
        pass

    def register_all_and_save(self, meshes_dir, landmarks_dir=None, **kwargs):
        self._show_results = False
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

            out_dir = os.path.join(meshes_dir, "registered")
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
    Procrustes registration. NICP is wrapping the trimesh implementations:
    trimesh.registration.nricp_amberg() and trimesh.registration.nricp_sumner().
    """
    def __init__(self, reference_path, reference_landmarks_path,
                 show_results=False):
        super().__init__(reference_path, reference_landmarks_path, show_results)
        self._p_registerer = ProcrustesLandmarkRegisterer(
            reference_path, reference_landmarks_path)
        self._reference_landmarks_barycentric = \
            utils.import_wrap_point_on_triangle_landmarks_as_b_coords(
                reference_landmarks_path
            )

    def register(self, mesh, mesh_landmarks=None, algorithm="sumner",
                 steps=None, eps=0.001, gamma=1, distance_threshold=0.1,
                 use_faces=False) -> trimesh.Trimesh:
        """
        Calling this function the reference mesh is fitted onto 'mesh'
        :param mesh: Trimesh to fit
        :param mesh_landmarks: landmarks of the mesh to fit
        :param algorithm: either amberg or sumner.
        :param steps:  Core parameters of the algorithm
            Iterable of iterables (wc, wi, ws, wl, wn).
            wc is the correspondence term (strength of fitting),
            wi is the identity term (recommended value : 0.001),
            ws is smoothness term, wl weights the landmark
            importance and wn the normal importance.
        :param eps: float. For amberg only. If the error decrease is inferior to
            this value, the current step ends.
        :param gamma: float. For amberg only. Weight the translation part
            against the rotational/skew part.
        :param distance_threshold: float. Distance threshold to account for
            a vertex match or not.
        :param use_faces: bool. If true, computes the closest distances
            also with respect to mesh faces. If false only with respect
            to vertices.

        :return: Trimesh obtained registering vertices of the reference mesh
            onto the target geometry.
        """
        landmark_aligned, mesh_landmarks = self._p_registerer.register(
            mesh, mesh_landmarks, return_landmarks=True)

        if algorithm == "amberg":
            if steps is None:
                steps = [
                    # ws, wl, wn, max_iter
                    [50, 5, 0.5, 3],
                    [20, 2, 0.5, 3],
                    [5, 0.5, 0.5, 3],
                    [0.5, 0, 0.5, 3],
                    [0.2, 0, 0.5, 3],
                ]

            registered_vertices = trimesh.registration.nricp_amberg(
                source_mesh=self._reference_mesh,
                target_geometry=landmark_aligned,
                source_landmarks=self._reference_landmarks_barycentric,
                target_positions=mesh_landmarks,
                steps=steps, eps=eps, gamma=gamma,
                distance_threshold=distance_threshold, use_faces=use_faces,
                use_vertex_normals=use_faces, return_records=False)

        elif algorithm == "sumner":
            if steps is None:
                wi, wn = 0.0001, 1
                steps = [
                    # wc, wi, ws, wl, wn
                    [0, wi, 50, 10, wn],
                    [0.1, wi, 20, 2, wn],
                    [1, wi, 5, 0.5, wn],
                    [5, wi, 2, 0.1, wn],
                    [10, wi, 0.8, 0, wn],
                    [50, wi, 0.5, 0, wn],
                    [100, wi, 0.35, 0, wn],
                    [1000, wi, 0.2, 0, wn],
                ]

            registered_vertices = trimesh.registration.nricp_sumner(
                source_mesh=self._reference_mesh,
                target_geometry=landmark_aligned,
                source_landmarks=self._reference_landmarks_barycentric,
                target_positions=mesh_landmarks,
                steps=steps, distance_threshold=distance_threshold,
                use_faces=use_faces, use_vertex_normals=use_faces,
                return_records=False)

        else:
            raise NotImplementedError(f"{algorithm} algorithm for NICP not "
                                      f"implemented yet. Use either 'amberg' "
                                      f"or 'sumner'.")

        registered_mesh = self._reference_mesh.copy()
        registered_mesh.vertices = registered_vertices

        if self._show_results:
            self.show_results(registered_mesh, comparison_mesh=landmark_aligned)
            registered_mesh.show()

        return registered_mesh


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
    registerer(mp, lp, algorithm="amberg")
    print("done")



