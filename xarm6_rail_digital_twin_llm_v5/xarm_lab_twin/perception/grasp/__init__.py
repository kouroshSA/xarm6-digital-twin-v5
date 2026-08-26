"""GG-CNN grasp detection on wrist-camera depth.

Depth in, ranked antipodal grasp candidates out, in the robot's world frame:

    from perception import SimWristCamera
    from perception.grasp import GGCNNDetector

    grasps = GGCNNDetector().detect(SimWristCamera(arm).capture())

Vendored from Douglas Morrison's GG-CNN (BSD-3) via UFACTORY's ufactory_vision
(BSD-3); see LICENSE.ggcnn, LICENSE.ufactory, and ggcnn.py's docstring for what
was changed and why. Needs torch, opencv, scipy and scikit-image, so the import
is deferred -- the sim itself does not require them.
"""

__all__ = ["GGCNNDetector", "Grasp"]


def __getattr__(name: str):
    if name in __all__:
        from .ggcnn import GGCNNDetector, Grasp
        return {"GGCNNDetector": GGCNNDetector, "Grasp": Grasp}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
