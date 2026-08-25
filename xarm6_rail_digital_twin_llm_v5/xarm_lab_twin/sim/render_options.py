"""Show or hide the lab backdrop.

The room behind the bench -- fume hood, reagent shelving, window, walls -- lives
in the ``lab_environment`` body, and every geom in it is in **MuJoCo geom group
2**. Nothing else in the scene uses that group, so the whole backdrop is one
visibility switch:

    from sim.render_options import scene_option

    renderer.update_scene(data, camera=cam, scene_option=scene_option(backdrop=False))

Why a geom group rather than two scene files, or an alpha flag
--------------------------------------------------------------
A second scene XML would be a second copy of everything else in it, which is
this repo's first standing defect class. Setting ``rgba`` alpha to 0 would hide
the backdrop from renders but leave it in the model, so ray casts and any
future point-cloud query would still hit a wall the picture says is not there.

A geom group is the one mechanism MuJoCo already applies consistently: the
renderer, the interactive viewer and ``mj_ray`` all take the same
``geomgroup`` mask, so "hidden" means hidden to all of them at once.

It also comes with a free control in the live viewer: **press ``2``** to toggle
the backdrop while the sim runs. No flag, no restart.

Turning it off restores exactly what the scene looked like before the backdrop
existed -- the bench against an empty void -- which is what makes it a fair
before/after rather than an approximation of one.
"""
from __future__ import annotations

import mujoco

# The group every lab_environment geom is assigned. Nothing else uses it, which
# is what makes the switch total rather than approximate.
BACKDROP_GROUP = 2


def scene_option(backdrop: bool = True,
                 base: mujoco.MjvOption | None = None) -> mujoco.MjvOption:
    """An ``MjvOption`` with the backdrop shown or hidden.

    ``base`` lets a caller layer this on top of options they already set;
    omitted, it starts from MuJoCo's defaults.
    """
    opt = base if base is not None else mujoco.MjvOption()
    opt.geomgroup[BACKDROP_GROUP] = 1 if backdrop else 0
    return opt


def ray_geomgroup(backdrop: bool = True):
    """The matching mask for ``mujoco.mj_ray``.

    Ray casting takes its own group mask, and it is easy to hide the backdrop
    from the renderer while still letting rays hit it -- which would put a
    surface in the depth map that the colour image says is not there. Deriving
    both masks from the same argument keeps them honest.
    """
    import numpy as np

    groups = np.ones(6, dtype=np.uint8)
    if not backdrop:
        groups[BACKDROP_GROUP] = 0
    return groups
