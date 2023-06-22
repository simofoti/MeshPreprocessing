import os
import numpy as np

import utils


class Normaliser:
    """ Normalisation class that can be very useful to prepare data for training
    in case they need to be normalised or it can be used to encrypt sensitive
    meshes (e.g. faces of real people). Obviously, if this class is used for
    encryption, the normalisation_dictionary behaves as the encryption key and
    needs to be transferred separately. All the normalised meshes can be
    considered to be encrypted. Anyone who visualise them will see random
    shapes unless they are un-normalised first.

    NB: meshes need to be in dense point correspondence. Run a non-rigid
    registration algorithm first
    (e.g. registration.ProcrustesLandmarkAndNicpRegisterer)"""

    def __init__(self, data_path, normalisation_dict_path=None,
                 random_normalisation_dict=False):
        self.data_path = data_path
        self._all_filenames = utils.find_filenames(self.data_path)

        self._out_dir = self.data_path + "_out"
        if not os.path.isdir(self._out_dir):
            os.mkdir(self._out_dir)

        if normalisation_dict_path is None:
            if random_normalisation_dict:
                v1 = self._load_mesh(self._all_filenames[0]).vertices
                normalisation_dict = {
                    'mean': np.random.random(v1.shape),
                    'std': np.random.random(v1.shape),
                }
            else:
                normalisation_dict = self._compute_normalisation_dict()
            self.mean = normalisation_dict['mean']
            self.std = normalisation_dict['std']
            np.save(os.path.join(self._out_dir, "norm.npy"), normalisation_dict)

        elif normalisation_dict_path.endswith('.pt'):
            import torch

            normalisation_dict = torch.load(normalisation_dict_path)
            self.mean = normalisation_dict['mean'].detach().cpu().numpy()
            self.std = normalisation_dict['std'].detach().cpu().numpy()
            np.save(normalisation_dict_path[:-3] + ".npy",
                    {'mean': self.mean, 'std': self.std})

        elif normalisation_dict_path.endswith('.npy'):
            normalisation_dict = np.load(normalisation_dict_path,
                                         allow_pickle=True)[()]
            self.mean = normalisation_dict['mean']
            self.std = normalisation_dict['std']

        else:
            raise NotImplementedError

    def __call__(self, mode="normalise"):
        for fname in self._all_filenames:
            mesh = self._load_mesh(fname)

            if mode == "normalise":
                mesh.vertices = self.normalise_verts(mesh.vertices)
            elif mode == "unnormalise":
                mesh.vertices = self.unnormalise_verts(mesh.vertices)
            else:
                raise NotImplementedError

            mesh.export(os.path.join(self._out_dir, fname))
        print(f"All meshes have been {mode}d and saved in {self._out_dir}.")

    def _load_mesh(self, filename):
        mesh_path = os.path.join(self.data_path, filename)
        return utils.load_trimesh(mesh_path)

    def _compute_normalisation_dict(self):
        train_verts = None
        for i, fname in enumerate(self._all_filenames):
            mesh_verts = self._load_mesh(fname).vertices
            if i == 0:
                train_verts = np.zeros(
                    [len(self._all_filenames), mesh_verts.shape[0], 3])
            train_verts[i, ::] = mesh_verts

        mean = np.mean(train_verts, axis=0)
        std = np.std(train_verts, axis=0)
        std = np.where(std > 0, std, 1e-8)
        return {'mean': mean, 'std': std}

    def normalise_verts(self, mesh_verts):
        return (mesh_verts - self.mean) / self.std

    def unnormalise_verts(self, mesh_verts):
        return mesh_verts * self.std + self.mean


if __name__ == '__main__':
    normaliser = Normaliser(
        data_path="/media/simo/DATASHURPRO/pre_post_fitted_meshes/encrypted",
        normalisation_dict_path="/media/simo/DATASHURPRO/pre_post_fitted_meshes/encrypted/norm.npy"
    )
    normaliser(mode="unnormalise")

