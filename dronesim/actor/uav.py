from panda3d.core import NodePath, Filename, Point3, TextNode
from .vehicle import VehicleModel
from dronesim.interface.control import IDroneControllable
from dronesim.utils import rad2deg
from dronesim.types import PandaFilePath, StateType
from typing import Union, Dict
from time import time
from direct.gui.OnscreenText import OnscreenText
import random


class UAVDroneModel(VehicleModel):
    '''
    Model class for a UAV Drone (quad, hex)-copter with separate Shell and propeller models.
    The model object can be synced to a state object via a IDroneControllable interface object.

    :param IDroneControllable update_source: Optional drone interface to update position/rotation with every update() call.
    If not specified, you can pass the state in the update(state).

    :param dict propellers indicates the set of bones to use in the model to attach propellers.
    the key is the name of the bone, and value is the model asset file name or NodePath object.
    The number of propellers would indicate the type of drone it is.

    :param dict propeller_spin indicates direction of spin for the given propeller bones, where value of
    '1' is clockwise, and '-1' is anti-clockwise. Direction is set to clockwise to any bones with unspecified direction.

    :param int drone_number: Optional drone identifier number, used in energy display.

    :param tuple energy_text_pos: (x, y) position of the energy display on screen.

    Default propeller layout is the 'Quad X' frame arrangement
    '''

    def __init__(self,
                 control_source: IDroneControllable,
                 shell_model: Union[PandaFilePath, NodePath] = None,
                 propellers: Dict[str, Union[PandaFilePath, NodePath]] = None,
                 propeller_spin: Dict[str, float] = None,
                 drone_number: int = None,
                 energy_text_pos: tuple = (-1.3, 0.9)):
        self._control_source = control_source
        self._drone_number = drone_number
        self._offset = Point3(0, 0, 0)  # will be added externally via set_offset

        if shell_model is None:
            shell_model = Filename("assets/models/quad-shell.glb")

        # Default propeller models
        if propellers is None:
            prop_model_cw = Filename("assets/models/propeller.glb")
            prop_model_ccw = prop_model_cw  # TODO: Temporary, create a flipped model

            propellers = {
                "PropellerJoint1": prop_model_ccw,
                "PropellerJoint2": prop_model_ccw,
                "PropellerJoint3": prop_model_cw,
                "PropellerJoint4": prop_model_cw
            }
            if propeller_spin is None:
                propeller_spin = {
                    "PropellerJoint1": -1,
                    "PropellerJoint2": -1,
                    "PropellerJoint3": 1,
                    "PropellerJoint4": 1
                }

        if propeller_spin is None:
            propeller_spin = dict()

        propeller_spin.update(
            {k: 1 for k in propellers.keys() if k not in propeller_spin})

        # Prefix so that it doesn't clash with original bone node
        propeller_parts = {'p_%s' % k: v for k, v in propellers.items()}

        self.joints = {'propellers': {}}

        super().__init__({
            'modelRoot': shell_model,
            **propeller_parts
        }, anims={'modelRoot': {}})  # To use the multipart w/o LOD loader

        for bone in propellers.keys():
            self.exposeJoint(None, 'modelRoot', bone)
            self.attach('p_%s' % bone, "modelRoot", bone)
            control_node = self.controlJoint(None, 'modelRoot', bone)

            self.joints['propellers'][bone] = {
                'bone': control_node,
                'spinDir': propeller_spin[bone]
            }
            control_node.setH(random.randint(0, 360))

        self._energy = 0.0  # Joules
        self._elapsed_time = 0.0  # Seconds
        self._power = 600  # Watts assumed constant power consumption
        self._last_update_time = 0.0
        self._energy_text_pos = energy_text_pos

        self._energy_text = OnscreenText(
            text=self._make_energy_text(0.0),
            pos=self._energy_text_pos,
            scale=0.07,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft
        )

        self.propellers_active = False

    def _make_energy_text(self, energy_val):
        if self._drone_number is not None:
            return f"Drone {self._drone_number} Energy: {energy_val:.2f} J"
        else:
            return f"Energy: {energy_val:.2f} J"

    def set_offset(self, x, y, z):
        self._offset = Point3(x, y, z)

    @property
    def controller(self) -> IDroneControllable:
        return self._control_source

    def update(self):
        current_time = time()
        if self._last_update_time == 0.0:
            self._last_update_time = current_time

        delta_time = current_time - self._last_update_time
        self._last_update_time = current_time

        # Check if UAV is airborne based on Z position
        pos = self.get_pos()
        is_airborne = pos.get_z() > 0.1

        if is_airborne:
            self._elapsed_time += delta_time
            self._energy = self._power * self._elapsed_time
            self.propellers_active = True
        else:
            self.propellers_active = False

        self._energy_text.setText(self._make_energy_text(self._energy))

        # Update model position/rotation from control source state
        state: StateType = self._control_source.get_current_state()
        if state is not None:
            state_info = state[3]
            transformState = state_info.get('state')
            if transformState is not None:
                base_pos = Point3(*transformState['pos'])
                self.setPos(base_pos + self._offset)

                rotx, roty, rotz = transformState['angle']
                self.setHpr(rad2deg(rotz), rad2deg(roty), rad2deg(rotx))
                thrust = transformState['thrust_vec']
                prop_vel = thrust.z * 1e5

                if self.propellers_active:
                    for bone in self.joints['propellers'].values():
                        bone['bone'].setH(bone['bone'].getH() + prop_vel * bone['spinDir'])

        super().update()
