import fire

import registration
import normalisation


def fire_no_out_print(component=None, command=None, name=None, serialize=None):
    try:
        fire.Fire(component, command, name, serialize)
    except ModuleNotFoundError as error_msg:
        if "open3d" in str(error_msg):
            # Fire tries to Print the trimesh result. Apparently this requires
            # additional libraries (i.e. open3d). Since the error is raised
            # only using the CLI and it does not affect the correct execution of
            # the code, the known error is detected and ignored.
            pass
        else:
            print(f"something went wrong wile fire was running the code"
                  f"the following error has been reported: {str(error_msg)}")


if __name__ == "__main__":

    registration_dict = {
        "procrustes_landmark_registration":
            registration.ProcrustesLandmarkRegisterer,
        "inertia_axes_and_icp_registration":
            registration.InertiaAxesAndIcpRegisterer,
        "procrustes_landmark_and_icp_registration":
            registration.ProcrustesLandmarkAndIcpRegisterer,
        "procrustes_landmark_and_nicp_registration":
            registration.ProcrustesLandmarkAndNicpRegisterer,
    }

    normalisation_dict = {
        "normalisation": normalisation.Normaliser,
        "normalization": normalisation.Normaliser,
        "encryption": normalisation.Normaliser
    }  # expose the same class with different names

    fire_no_out_print({**registration_dict, **normalisation_dict})
