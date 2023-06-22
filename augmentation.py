import os

import scipy.sparse.csgraph
import trimesh
import numpy as np

from abc import ABC, abstractmethod
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian

import utils


class MeshAugmentation(ABC):
    """Abstract class for mesh augmentation algorithms.
    The template mesh is not mandatory, but it is useful when augmenting
    multiple meshes because the eigendecomposition of the mesh laplacian is
    computed only once. """

    def __init__(self, template_path=None, number_eigenvectors=500,
                 show_results=False):
        self._number_eigenvectors = number_eigenvectors
        self._show_results = show_results

        if template_path is None:
            self._template, self._eigvecs = None, None
        else:
            self._template = utils.load_trimesh(template_path)
            self._eigvecs = self._compute_eigenvectors(self._template,
                                                       number_eigenvectors)

    def __call__(self, mesh_path, mesh2_path=None, **kwargs):
        mesh = utils.load_trimesh(mesh_path)

        # if augmentation requires 2 meshes, load also the second mesh
        if mesh2_path is not None:
            kwargs["mesh2"] = utils.load_trimesh(mesh2_path)

        return self.augment(mesh, **kwargs)

    @abstractmethod
    def augment(self, mesh, **kwargs) -> trimesh.Trimesh:
        pass

    def augment_all_and_save(self, meshes_dir, augmentation_factor=3, **kwargs):
        assert augmentation_factor > 1
        self._show_results = False
        out_dir = os.path.join(meshes_dir, "augmented")

        all_files = utils.find_filenames(meshes_dir)

        for mesh_path in all_files:
            for i in range(augmentation_factor):
                selector = np.random.choice(len(all_files), 1, replace=False)
                mesh2_path = all_files[selector]  # not necessarily used
                aug_mesh = self.__call__(mesh_path=mesh_path,
                                         mesh2_path=mesh2_path,
                                         **kwargs)
                mesh_name = os.path.split(mesh_path)[1]
                mesh_name_split = mesh_name.split('.')
                aug_mesh_name = mesh_name_split[0] + \
                    f"_aug{i}." + mesh_name_split[1]
                aug_mesh.export(os.path.join(out_dir, aug_mesh_name))

    @staticmethod
    def _compute_eigenvectors(mesh, number_eigenvectors):
        # from face_to_edge
        face = mesh.faces.T
        edge_index = np.concatenate([face[:2], face[1:], face[::2]], axis=1)
        row, col = edge_index
        row, col = np.concatenate([row, col]), np.concatenate([col, row])
        edge_index = np.stack([row, col], axis=0)

        # coalesce -> row-wise sorts and remove duplicate entries
        sort_by_row = True
        nnz = edge_index.shape[1]
        num_nodes = mesh.vertices.shape[0]
        idx = np.empty(nnz + 1, dtype=int)
        idx[0] = -1
        idx[1:] = edge_index[1 - int(sort_by_row)]
        idx[1:] = idx[1:] * num_nodes + edge_index[int(sort_by_row)]
        tmp_idx1 = idx[1:]
        perm = np.argsort(tmp_idx1)
        idx[1:] = tmp_idx1[perm]

        edge_index = edge_index[:, perm]
        mask = idx[1:] > idx[:-1]
        edge_index = edge_index[:, mask]

        r, c = edge_index[0, :], edge_index[1, :]
        adj = scipy.sparse.coo_matrix((np.ones_like(r), (r, c)),
                                      shape=(num_nodes, num_nodes))
        lapl = laplacian(adj).astype('float')

        _, eigvecs = eigsh(lapl, k=number_eigenvectors, which='SM')
        return eigvecs

    @staticmethod
    def show_results(original_mesh, result_mesh, original_mesh2=None):
        scene = trimesh.Scene()
        scene.add_geometry(result_mesh)

        scale = result_mesh.scale
        original_mesh.vertices[:, 0] -= scale
        scene.add_geometry(original_mesh)

        if original_mesh2 is not None:
            original_mesh2.vertices[:, 0] += scale
            scene.add_geometry(original_mesh2)
        scene.show()


class RandomSpectralPerturbation(MeshAugmentation):
    """ With this augmentation technique the spectral components of a mesh are
    randomly perturbed."""
    def augment(self, mesh, lowest_frequency_altered=4,
                number_of_frequencies_to_alter=3,
                min_perturbation=-2, max_perturbation=2, **kwargs):
        """
        Augmentation function. The spectral components of the mesh are obtained
        eigendecomposing the Kirchoff Laplacian of the mesh and multiplying
        the vertices by the eigenvectors. The eigenvectors of the Laplacian can
        be considered as Fourier bases. Therefore, the vertices are Fourier
        transformed, the result is perturbed, and antitransformed back to the
        vertex domain.

        :param mesh: mesh to be augmented.
        :param lowest_frequency_altered: lowest frequency to alter. As the
            lowest frequencies are associated to low frequency shape changes
            like scaling across the main axes, it may be beneficial to preserve
            the first 3-4 values.
        :param number_of_frequencies_to_alter: number of frequencies to alter.
            The more frequencies are altered, the more changes we might expect.
        :param min_perturbation: lower bound of the perturbation, which is
            performed sampling a uniform distribution.
        :param max_perturbation: upper bound of the perturbation, which is
            performed sampling a uniform distribution.
        :param kwargs: additional arguments.

        :return: augmented mesh.
        """

        if self._template is None and self._eigvecs is None:
            self._eigvecs = self._compute_eigenvectors(
                mesh, self._number_eigenvectors)

        indices = np.random.randint(
            lowest_frequency_altered, self._number_eigenvectors - 1,
            size=self._number_eigenvectors - number_of_frequencies_to_alter - 1)

        deform_v = np.ones(self._number_eigenvectors)
        deform_v[indices] = np.random.uniform(
            min_perturbation, max_perturbation,
            self._number_eigenvectors - number_of_frequencies_to_alter - 1)
        deform_m = np.diag(deform_v)

        v_deformed = self._eigvecs @ deform_m @ self._eigvecs.T @ mesh.vertices

        augmented_mesh = mesh.copy()
        augmented_mesh.vertices = v_deformed

        if self._show_results:
            self.show_results(mesh, augmented_mesh)

        return augmented_mesh


class RandomLinearInterpolation(MeshAugmentation):
    """ With this augmentation technique, new meshes are created by linearly
    interpolating the vertex coordinates of two meshes. The amount of the
    interpolation is the same for all vertices."""

    def augment(self, mesh, mesh2=None, **kwargs):
        """
        Augmentation function. new meshes are created by linearly
        interpolating the vertex coordinates of two meshes.

        Note that the template that may have been provided during initialisation
        does not influence the interpolation. All meshes should have the same
        topology and number of vertices. You may need to register them with
        nicp before augmenting them.

        :param mesh: first mesh
        :param mesh2: second mesh
        :param kwargs: additional arguments
        :return: augmented mesh
        """
        assert mesh2 is not None

        interpolation_value = np.random.uniform(size=1).item()
        v_interpolated = utils.interpolate(mesh.vertices, mesh2.vertices,
                                           interpolation_value)

        augmented_mesh = mesh.copy()
        augmented_mesh.vertices = v_interpolated

        if self._show_results:
            self.show_results(mesh, augmented_mesh, mesh2)

        return augmented_mesh


class RandomSpectralInterpolation(MeshAugmentation):
    """ With this augmentation technique, new meshes are created by
        interpolating the spectral components of two meshes. The amount of the
        interpolation can differ across components."""

    def augment(self, mesh, mesh2=None,
                interpolate_frequencies_until=30, **kwargs):
        """
        Augmentation function. The spectral components of the two meshes are
        obtained multiplying the vertex coordinated by the eigenvalues of the
        Kirchoff Laplacian operator. The new spectral components are obtained as
        a random interpolation of the spectra obtained from the input meshes.
        The new spectral components are antitransformed to obtain the new
        vertices.
        Note that the template that may have been provided during initialisation
        does not influence the interpolation. All meshes should have the same
        topology and number of vertices. You may need to register them with
        nicp before augmenting them.

        :param mesh: first mesh
        :param mesh2: second mesh
        :param interpolate_frequencies_until:
        :param kwargs: additional arguments
        :return: augmented mesh
        """
        assert mesh2 is not None

        if self._template is None and self._eigvecs is None:
            self._eigvecs = self._compute_eigenvectors(
                mesh, self._number_eigenvectors)

        s1 = self._eigvecs.T @ mesh.vertices
        s2 = self._eigvecs.T @ mesh2.vertices

        values = np.random.normal(loc=0.5, scale=0.5, size=[s1.shape[0], 1])
        s3 = s1 + values * (s2 - s1)

        s4 = s1.copy()
        s4[:interpolate_frequencies_until] = s3[:interpolate_frequencies_until]

        augmented_mesh = mesh.copy()
        augmented_mesh.vertices = self._eigvecs @ s4

        if self._show_results:
            self.show_results(mesh, augmented_mesh, mesh2)

        return augmented_mesh


if __name__ == "__main__":
    m1p = "/media/simo/DATASHURPRO/pre_post_fitted_meshes/original/a_22.obj"
    m2p = "/media/simo/DATASHURPRO/pre_post_fitted_meshes/original/a_35.obj"

    augmenter = RandomSpectralInterpolation(m1p, number_eigenvectors=500,
                                            show_results=True)
    augmenter(m1p, m2p)

    augmenter = RandomLinearInterpolation(show_results=True)
    augmenter(m1p, m2p)

    augmenter = RandomSpectralPerturbation(m1p, number_eigenvectors=500,
                                           show_results=True)
    augmenter(m1p, lowest_frequency_altered=4,
              number_of_frequencies_to_alter=3,
              min_perturbation=0.8, max_perturbation=1.2)


