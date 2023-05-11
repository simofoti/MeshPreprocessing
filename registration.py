import trimesh
import os.path

import numpy as np

from abc import ABC, abstractmethod
from scipy.linalg import orthogonal_procrustes

import utils


class Registerer(ABC):
    def __init__(self, reference_path, reference_landmarks_path=None):
        self._reference_mesh = utils.load_trimesh(reference_path)
        if reference_landmarks_path is not None:
            self._reference_landmarks = utils.load_landmarks(
                reference_landmarks_path, mesh=self._reference_mesh)
        else:
            self._reference_landmarks = None

    @abstractmethod
    def register(self, mesh_path,
                 mesh_landmarks_path=None, **kwargs) -> trimesh.Trimesh:
        pass

    def register_all_and_save(self, meshes_dir, landmarks_dir=None, **kwargs):
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

            registered_mesh = self.register(fname, lms_path, **kwargs)

            out_dir = os.path.join(meshes_dir, "registered")
            registered_mesh.export(
                os.path.join(out_dir, os.path.split(fname)[1]))


class ProcrustesLandmarkRegisterer(Registerer):
    def register(self, mesh_path, mesh_landmarks_path=None, show_results=False):
        assert mesh_landmarks_path is not None
        assert self._reference_landmarks is not None
        mesh = utils.load_trimesh(mesh_path)
        lms = utils.load_landmarks(mesh_landmarks_path, mesh=mesh)

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

        # updated landmarks can be obtained with vvvvv
        # reg_m_lms = np.dot(centered_lms, rotation.T) * scale
        # reg_m_lms = (reg_m_lms * norm_ref_lms) + translation_ref

        registered_mesh = mesh.copy()
        registered_mesh.vertices = reg_m_verts

        if show_results:
            scene = trimesh.Scene()
            scene.add_geometry(self._reference_mesh)
            scene.add_geometry(
                trimesh.points.PointCloud(self._reference_landmarks))
            scene.add_geometry(registered_mesh)
            scene.show()

        return registered_mesh


if __name__ == "__main__":
    rmp = "/home/simo/Desktop/bws_project/template/template_sym2_fixed.ply"
    rlp = "/home/simo/Desktop/bws_project/template/template_sym2_fixed_ibug_68.txt"

    mp = "/media/simo/DATASHURPRO/SD-VAE Frontofacial Outcomes/Pilot Data/Meshes/1/626491 Post-Op STL.stl"
    lp = "/media/simo/DATASHURPRO/SD-VAE Frontofacial Outcomes/Pilot Data/Landmarks/1/626491 Post-Op Landmarks.txt"

    registerer = ProcrustesLandmarkRegisterer(reference_path=rmp,
                                              reference_landmarks_path=rlp)
    registerer.register(mp, lp, show_results=True)
    print("done")



