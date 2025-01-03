__author__ = 'yuwenhao'

import numpy as np
#from policy_transfer.envs.dart import dart_env

##############################################################################################################
################################  Hopper #####################################################################
##############################################################################################################


class mjHopperManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.2, 1.0]  # friction range
        self.mass_range = [2.0, 20.0]
        self.damping_range = [0.15, 2.0]
        self.power_range = [150, 500]
        self.velrew_weight_range = [-1.0, 1.0]
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.ankle_jnt_range = [0.5, 1.0]

        self.activated_param = [0, 1,2,3,4, 5,6,7, 8, 10, 11, 12, 13]
        self.controllable_param = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        mass_param = []
        for bid in range(1, 5):
            cur_mass = self.simulator.model.body_mass[bid]
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        damp_param = []
        for jid in range(3, 6):
            cur_damp = self.simulator.model.dof_damping[jid]
            damp_param.append((cur_damp - self.damping_range[0]) / (self.damping_range[1] - self.damping_range[0]))

        cur_power = self.simulator.model.actuator_gear[0][0]
        power_param = (cur_power - self.power_range[0]) / (self.power_range[1] - self.power_range[0])

        cur_velrew_weight = self.simulator.velrew_weight
        velrew_param = (cur_velrew_weight - self.velrew_weight_range[0]) / (
                self.velrew_weight_range[1] - self.velrew_weight_range[0])

        cur_restitution = self.simulator.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.simulator.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.simulator.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.simulator.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        cur_jntlimit = self.simulator.model.jnt_range[-1][0]
        jntlimit_param = (cur_jntlimit - self.ankle_jnt_range[0]) / (self.ankle_jnt_range[1] - self.ankle_jnt_range[0])

        params = np.array([friction_param] + mass_param + damp_param + [power_param, velrew_param, rest_param
                                                                        ,solimp_param, solref_param, armature_param,
                                                                        jntlimit_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.model.geom_friction[-1][0] = friction
            cur_id += 1
        for bid in range(1, 5):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.model.body_mass[bid] = mass
                cur_id += 1
        for jid in range(5, 8):
            if jid in self.controllable_param:
                damp = x[cur_id] * (self.damping_range[1] - self.damping_range[0]) + self.damping_range[0]
                self.simulator.model.dof_damping[jid - 2] = damp
                cur_id += 1
        if 8 in self.controllable_param:
            power = x[cur_id] * (self.power_range[1] - self.power_range[0]) + self.power_range[0]
            self.simulator.model.actuator_gear[0][0] = power
            self.simulator.model.actuator_gear[1][0] = power
            self.simulator.model.actuator_gear[2][0] = power
            cur_id += 1
        if 9 in self.controllable_param:
            velrew_weight = x[cur_id] * (self.velrew_weight_range[1] - self.velrew_weight_range[0]) + \
                            self.velrew_weight_range[0]
            self.simulator.velrew_weight = velrew_weight
            cur_id += 1
        if 10 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 11 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.simulator.model.geom_solimp)):
                self.simulator.model.geom_solimp[bn][0] = solimp
                self.simulator.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 12 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 13 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 6):
                self.simulator.model.dof_armature[dof] = armature
            cur_id += 1
        if 14 in self.controllable_param:
            jntlimit = x[cur_id] * (self.ankle_jnt_range[1] - self.ankle_jnt_range[0]) + \
                            self.ankle_jnt_range[0]
            self.simulator.model.jnt_range[-1][0] = -jntlimit
            self.simulator.model.jnt_range[-1][1] = jntlimit
            cur_id += 1

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class mjWalkerParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.mass_range = [2.0, 15.0]
        self.range = [0.5, 2.0]  # friction range
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.tilt_z_range = [-0.18, 0.18]

        self.activated_param = [0]
        self.controllable_param = [0]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        mass_param = []
        for bid in range(1, 8):
            cur_mass = self.simulator.model.body_mass[bid]
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        cur_friction = self.simulator.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_restitution = self.simulator.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.simulator.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.simulator.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.simulator.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        cur_tiltz = self.simulator.tilt_z
        tiltz_param = (cur_tiltz - self.tilt_z_range[0]) / (self.tilt_z_range[1] - self.tilt_z_range[0])

        params = np.array(mass_param + [friction_param, rest_param ,solimp_param, solref_param, armature_param, tiltz_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        for bid in range(0, 7):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.simulator.model.body_mass[bid] = mass
                cur_id += 1

        if 7 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.model.geom_friction[-1][0] = friction
            cur_id += 1
        if 8 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 9 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.simulator.model.geom_solimp)):
                self.simulator.model.geom_solimp[bn][0] = solimp
                self.simulator.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 10 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 11 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 6):
                self.simulator.model.dof_armature[dof] = armature
            cur_id += 1
        if 12 in self.controllable_param:
            tiltz = x[cur_id] * (self.tilt_z_range[1] - self.tilt_z_range[0]) + self.tilt_z_range[0]
            self.simulator.tilt_z = tiltz
            self.simulator.model.opt.gravity[:] = [9.81 * np.sin(tiltz), 0.0, -9.81 * np.cos(tiltz)]
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)


class mjcheetahParamManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.2, 1.0]  # friction range
        self.velrew_weight_range = [-1.0, 1.0]
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.tilt_z_range = [-0.18, 0.18]

        self.activated_param = [5]
        self.controllable_param = [5]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_velrew_weight = self.simulator.velrew_weight
        velrew_param = (cur_velrew_weight - self.velrew_weight_range[0]) / (
                self.velrew_weight_range[1] - self.velrew_weight_range[0])

        cur_restitution = self.simulator.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.simulator.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.simulator.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.simulator.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        cur_tiltz = self.simulator.tilt_z
        tiltz_param = (cur_tiltz - self.tilt_z_range[0]) / (self.tilt_z_range[1] - self.tilt_z_range[0])

        params = np.array([friction_param, velrew_param, rest_param ,solimp_param, solref_param, armature_param, tiltz_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        if 0 in self.controllable_param:
            friction = x[cur_id] * (self.range[1] - self.range[0]) + self.range[0]
            self.simulator.model.geom_friction[-1][0] = friction
            cur_id += 1
        if 1 in self.controllable_param:
            velrew_weight = x[cur_id] * (self.velrew_weight_range[1] - self.velrew_weight_range[0]) + \
                            self.velrew_weight_range[0]
            self.simulator.velrew_weight = velrew_weight
            cur_id += 1
        if 2 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 3 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.simulator.model.geom_solimp)):
                self.simulator.model.geom_solimp[bn][0] = solimp
                self.simulator.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 4 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.simulator.model.geom_solref)):
                self.simulator.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 5 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 6):
                self.simulator.model.dof_armature[dof] = armature
            cur_id += 1
        if 6 in self.controllable_param:
            tiltz = x[cur_id] * (self.tilt_z_range[1] - self.tilt_z_range[0]) + self.tilt_z_range[0]
            self.simulator.tilt_z = tiltz
            self.simulator.model.opt.gravity[:] = [9.81 * np.sin(tiltz), 0.0, -9.81 * np.cos(tiltz)]
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)
