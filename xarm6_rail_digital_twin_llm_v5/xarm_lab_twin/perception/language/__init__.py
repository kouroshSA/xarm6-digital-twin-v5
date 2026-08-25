"""Language-conditioned targeting for the wrist camera.

Say what you want; get the grasp for it:

    from perception import SimWristCamera
    from perception.language import LanguageTargeter

    grasp = LanguageTargeter().grasp_for("the blue cube",
                                         SimWristCamera(arm).capture())

Grounding DINO proposes regions from free text, depth turns each region into a
measured physical object, and GG-CNN supplies the grasp. See targeter.py's
docstring for why the depth stage is not optional -- briefly, an image carries
no scale, so "a green cube" grounds equally well to the green bin.

Needs transformers + torch; imported lazily so the sim does not require them.
"""

__all__ = ["Detection", "GroundingDINOGrounder", "LanguageTargeter", "Target"]


def __getattr__(name: str):
    if name in ("LanguageTargeter", "Target"):
        from .targeter import LanguageTargeter, Target
        return {"LanguageTargeter": LanguageTargeter, "Target": Target}[name]
    if name in ("Detection", "GroundingDINOGrounder"):
        from .grounding import Detection, GroundingDINOGrounder
        return {"Detection": Detection,
                "GroundingDINOGrounder": GroundingDINOGrounder}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
