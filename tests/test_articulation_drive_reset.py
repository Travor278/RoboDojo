"""Exercise the vendored articulation methods with real USD and a fake simulator.

Loading the class AST avoids importing Isaac Sim, IsaacLab or Torch. Only the
simulator boundary is replaced; drive discovery/capture/reset use usd-core.
"""

import ast
import random
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from robodojo_runtime import source_root


@pytest.fixture
def articulation():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/button")
    events = []

    class SingleArticulation:
        def initialize(self, *, physics_sim_view):
            events.append("initialize")

    source = source_root() / "env/scene_manager/objects/articulation.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ArticulationObject"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            cls,
        ],
        type_ignores=[],
    )
    app = SimpleNamespace(update=lambda: events.append("update"))
    namespace = {
        "SingleArticulation": SingleArticulation,
        "SimulationManager": SimpleNamespace(get_physics_sim_view=lambda: object()),
        "omni": SimpleNamespace(kit=SimpleNamespace(app=SimpleNamespace(get_app=lambda: app))),
        "get_prim_at_path": stage.GetPrimAtPath,
        "Usd": Usd,
        "UsdGeom": UsdGeom,
        "UsdPhysics": UsdPhysics,
        "np": np,
        "random": random,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)  # noqa: S102
    obj = object.__new__(namespace["ArticulationObject"])
    obj._prim_path = "/button"
    obj.prim = stage.GetPrimAtPath("/button")
    obj.dof_properties = {"upper": np.array([0.005]), "lower": np.array([-0.005])}
    obj.get_current_joint_positions = lambda: np.array([0.002])
    obj.set_current_joint_positions = lambda positions: events.append("positions")
    obj.default_pos = (0, 0, 0)
    obj.default_ori = (1, 0, 0, 0)
    obj.init_scale = (1, 1, 1)
    obj.set_local_pose = lambda pos, ori: events.append("pose")
    obj.set_local_scale = lambda scale: None
    obj._apply_default_velocities = lambda: None
    obj.visual_cfg = {}
    obj.physics_cfg = {"reset_drive": True}
    return obj, stage, events


def add_drive(stage, name, joint_type, drive_type, target):
    joint = joint_type.Define(stage, f"/button/{name}")
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_type)
    drive.CreateTargetPositionAttr(target)
    return joint.GetPrim(), drive


def test_initialize_captures_authored_targets_and_reset_restores_them(articulation):
    obj, stage, _ = articulation
    _, spring = add_drive(stage, "spring", UsdPhysics.PrismaticJoint, "linear", 0.006)
    _, hinge = add_drive(stage, "hinge", UsdPhysics.RevoluteJoint, "angular", 90.0)
    obj.initialize()
    assert obj.initial_drive_targets == {
        "/button/spring": pytest.approx(0.006),
        "/button/hinge": 90.0,
    }
    spring.GetTargetPositionAttr().Set(-0.005)
    hinge.GetTargetPositionAttr().Set(15.0)
    obj.reset_drive_targets()
    assert spring.GetTargetPositionAttr().Get() == pytest.approx(0.006)
    assert hinge.GetTargetPositionAttr().Get() == 90.0
    # Repeated resets reuse the original snapshot, not the modified drive state.
    spring.GetTargetPositionAttr().Set(0.0)
    obj.reset_drive_targets()
    assert spring.GetTargetPositionAttr().Get() == pytest.approx(0.006)


@pytest.mark.parametrize("method", ["apply_saved_pose", "relocate_offscreen"])
@pytest.mark.parametrize("physics_cfg", [{"reset_drive": True}, {"reset_drive": False}, {}])
def test_pose_reset_honors_marker_after_restoring_positions(articulation, method, physics_cfg):
    obj, stage, events = articulation
    _, spring = add_drive(stage, "spring", UsdPhysics.PrismaticJoint, "linear", 0.006)
    obj.initialize()
    spring.GetTargetPositionAttr().Set(-0.005)
    obj.physics_cfg = physics_cfg
    original = obj.reset_drive_targets

    def restore():
        events.append("drive")
        original()

    obj.reset_drive_targets = restore
    events.clear()
    getattr(obj, method)()
    if physics_cfg.get("reset_drive", False):
        assert events[:3] == ["positions", "drive", "pose"]
        assert spring.GetTargetPositionAttr().Get() == pytest.approx(0.006)
    else:
        assert "drive" not in events
        assert spring.GetTargetPositionAttr().Get() == pytest.approx(-0.005)


def test_capture_ignores_non_joints_and_joints_without_drives(articulation):
    obj, stage, _ = articulation
    UsdPhysics.PrismaticJoint.Define(stage, "/button/no_drive")
    prim = UsdGeom.Xform.Define(stage, "/button/not_a_joint").GetPrim()
    UsdPhysics.DriveAPI.Apply(prim, "linear").CreateTargetPositionAttr(2.0)
    _, drive = add_drive(stage, "generic", UsdPhysics.Joint, "angular", 45.0)
    assert obj._capture_drive_targets() == {"/button/generic": 45.0}
    drive.GetTargetPositionAttr().Set(50.0)
    assert obj._capture_drive_targets() == {"/button/generic": 50.0}


def test_capture_rejects_invalid_articulation_root(articulation):
    obj, _, _ = articulation
    obj._prim_path = "/missing"
    with pytest.raises(ValueError, match="Invalid articulation root path: /missing"):
        obj._capture_drive_targets()


def test_reset_skips_removed_joints_and_drives(articulation):
    obj, stage, _ = articulation
    removed, _ = add_drive(stage, "removed", UsdPhysics.PrismaticJoint, "linear", 0.006)
    no_drive, _ = add_drive(stage, "no_drive", UsdPhysics.RevoluteJoint, "angular", 90.0)
    _, retained = add_drive(stage, "retained", UsdPhysics.PrismaticJoint, "linear", 0.004)
    obj.initialize()
    stage.RemovePrim(removed.GetPath())
    no_drive.RemoveAPI(UsdPhysics.DriveAPI, "angular")
    retained.GetTargetPositionAttr().Set(0.0)
    obj.reset_drive_targets()
    assert retained.GetTargetPositionAttr().Get() == pytest.approx(0.004)


def test_reset_recreates_removed_target_attribute(articulation):
    obj, stage, _ = articulation
    prim, drive = add_drive(stage, "spring", UsdPhysics.PrismaticJoint, "linear", 0.006)
    obj.initialize()
    prim.RemoveProperty(drive.GetTargetPositionAttr().GetName())
    obj.reset_drive_targets()
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(0.006)
