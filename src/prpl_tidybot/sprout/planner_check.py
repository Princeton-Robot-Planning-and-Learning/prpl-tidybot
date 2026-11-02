"""Offline dependency check for the shared PRPL URDF planning backend."""

import argparse

from pybullet_helpers.ikfast.load import import_ikfast

from prpl_tidybot.sprout.urdf_collision import PRPLCollisionScene, urdf_digest


def check(*, build_ikfast: bool = False) -> None:
    """Load model/collision assets and optionally prepare the workstation solver."""
    with PRPLCollisionScene() as scene:
        if build_ikfast:
            info = scene.arm.ikfast_info()
            if info is None:
                raise RuntimeError("PRPL Kinova model has no IKFast configuration")
            print(
                "Checking PRPL Kinova IKFast (first setup may compile C++)...",
                flush=True,
            )
            try:
                import_ikfast(info)
            except (RuntimeError, ImportError) as error:
                raise RuntimeError(
                    "IKFast setup failed. On the Linux workstation install "
                    "build-essential liblapack-dev libblas-dev libgfortran5, "
                    "then rerun setup. Nonstandard library paths use LAPACK_DIR, "
                    "BLAS_DIR and LIBGFORTRAN_DIR from pybullet_helpers."
                ) from error
        print("PRPL URDF collision assets OK: " + urdf_digest(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-ikfast", action="store_true")
    check(build_ikfast=parser.parse_args().build_ikfast)
