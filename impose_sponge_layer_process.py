"""
Sponge Layer outflow condition for FluidDynamics in KratosMultiphysics

"""

######################################################################
# Chair of Structural Analysis - TUM (Statik)
# Author: Ammar Khallouf
# Date: 04.05.21
# Version: 1.0
######################################################################
""" 
*** Description ***

This process applies a selective damping force to absorb velocity fluctuations
within a specfic region of the domain (e.g near outlet boundary)

The damping force takes the form [ F= -k(u-u_mean) ]
"""
######################################################################
"""
*** References ***

Khallouf, A: Developments for Improved FEM-Based LES Simulations in Computational Wind Engineering (Master's Thesis)(2021)

Wei, Gengsheng: The sponge layer method in flow-3d. In: Flow Science, Santa Fe, NM (2015)

"""
######################################################################

"""
Example of usage for user-defined exponential profile:

Required Input:

1. Volume model part for the fluid domain

2. Exponential mean velocity profile parameters: ([z_ref: reference height] , [u_mean: mean velocity at z_ref], [alpha: profile exponent]) 

3. Damping coefficent k (recommended value = 1.0)

4. 3D Extents of the sponge layer [x_start, x_end] , [y_start, y_end] , [z_start, z_end]
   with: (x_start > x_end) , (y_start > y_end) , (z_start > z_end)

{
    "python_module" : "sponge_layer_process",
    "Parameters"    : {
        "fluid_model_part_name" : "FluidModelPart.ModelPartName",
        "print_to_screen": false,
        "x_start" : 1200,
        "x_end"   : 1800,
        "y_start" : -450,
        "y_end"   : 450,
        "z_start" : 0.0,
        "z_end"   : 600,
        "damping_coefficent"   : 1.00,
        "profile_type"    : "ud_exponential",
        "u_mean"           : 40.0,
        "z_ref"           : 70,
        "alpha"           : 0.25
    }
}
"""
######################################################################
"""
Example of usage for logarithmic profile:

1. Volume model part for the fluid domain

2. Logarithmic mean velocity profile parameters: ([z_0: roughness length > 0] , [u_friction: frictional (shear) velocity]) 

3. Damping coefficent k (recommended value = 1.0)

4. 3D Extents of the sponge layer [x_start, x_end] , [y_start, y_end] , [z_start, z_end]

   with: (x_start < x_end) , (y_start < y_end) , (z_start < z_end)

{
    "python_module" : "sponge_layer_process",
    "Parameters"    : {
        "inlet_model_part_name" : "FluidModelPart.ModelPartName",
        "print_to_screen": false,
        "x_start" : 1200,
        "x_end"   : 1800,
        "y_start" : -450,
        "y_end"   : 450,
        "z_start" : 0.0,
        "z_end"   : 600,
        "damping_coefficent"   : 1.00,
        "profile_type"    : "logarithmic",
        "u_friction"      : 40.0,
        "z_0"             :0.00001

    }
}
"""
######################################################################

__all__ = ["Factory", "ImposeSpongeLayerProcess"]

import os

from collections import namedtuple, Mapping
from math import isclose, floor, ceil, log

from KratosMultiphysics import VELOCITY_X, VELOCITY_Y, VELOCITY_Z
from KratosMultiphysics import BODY_FORCE_X, BODY_FORCE_Y, BODY_FORCE_Z
from KratosMultiphysics import TIME, DELTA_TIME, NodesArray
from KratosMultiphysics import Model, Logger

# other imports
from KratosMultiphysics.time_based_ascii_file_writer_utility import (
    TimeBasedAsciiFileWriterUtility,
)
from KratosMultiphysics import Parameters as KM_Parameters
from KratosMultiphysics import IS_RESTARTED as KM_IS_RESTARTED


class Parameters(Mapping):
    def __init__(self, kratos_parameters):
        self._kratos_parameters = kratos_parameters

    def __getitem__(self, key):
        param = self._kratos_parameters[key]
        if param.IsDouble():
            value = param.GetDouble()
        elif param.IsInt():
            value = param.GetInt()
        elif param.IsString():
            value = param.GetString()
        elif param.IsBool():
            value = param.GetBool()
        else:
            value = param
        return value

    def __iter__(self):
        yield from self._kratos_parameters.keys()

    def __len__(self):
        return self._kratos_parameters.size()


def Factory(settings, Model):
    return ImposeSpongeLayerProcess(Model, Parameters(settings["Parameters"]))


# -------------------------------Logarithmic Velcity Profile-----------------------------------
class LogMeanProfile:
    def __init__(self, friction_velocity, z_0, key=lambda node: node.Z):
        self.friction_velocity = friction_velocity
        self.z_0 = (
            z_0 + 1.0e-07
        )  # avoid division by zero for roughness length (i.e z_0>0.0)
        self.key = key

    def mean_wind_speed(self, node, von_karman_const=0.41):
        return (
            1
            / von_karman_const
            * self.friction_velocity
            * log((self.key(node)) / self.z_0)
        )


# -------------------------------Exponential Velcity Profile-----------------------------------
class UdExpMeanProfile:
    def __init__(self, u_mean, z_ref, alpha, key=lambda node: node.Z):
        self.key = key
        self.u_mean = u_mean
        self.z_ref = z_ref
        self.alpha = alpha

    def mean_wind_speed(self, node):
        return self.u_mean * (self.key(node) / self.z_ref) ** self.alpha


# -------------------------------Sponge Layer Utility-----------------------------------


class ImposeSpongeLayerProcess:
    @property
    def sponge_nodes(self):

        # Store the nodes located inside the spone layer in Kratos array

        self.Sponge_layer_nodes = NodesArray()
        for node in self.model_part.Nodes:
            if (
                self.x_end >= node.X >= self.x_start
                and self.y_end >= node.Y >= self.y_start
                and self.z_end >= node.Z >= self.z_start
            ):
                # print(node.X)
                self.Sponge_layer_nodes.append(node)

        # Raise error if no nodes are selected for sponge layer (i.e empty array for sponge nodes)

        if not (self.Sponge_layer_nodes):
            raise Exception(
                "Error: Sponge layer contains no nodes,  Check layer starting and ending coordinates x,y,z"
            )

        return self.Sponge_layer_nodes

    #########################################

    def __init__(self, Model, settings):

        # Read input settings form .json file

        for name, value in settings.items():

            setattr(self, name, value)

        # Read volume model part name

        self.model_part = Model[self.fluid_model_part_name]

        # Damping factors at start and end of the sponge layer

        self.k_0 = 0.0
        self.k_1 = self.damping_coefficent

        # Sponge layer streamwise thickness

        self.d_ = self.x_end - self.x_start

        # Damping force mutiplier

        self.F_k = self.k_0 + ((self.k_1 - self.k_0) / self.d_)

        # Handle exceptions for reading velocity profile

        if self.profile_type == "logarithmic":
            self.mean_profile = self.CreateLogMeanProfile()

            Logger.PrintInfo(
                "\n## Sponge Layer Condition Summary ##",
                "\nImposeSpongeLayerProcess: ",
                "Adopting mean logarithmic velocity profile with: ",
                "u_friction =",
                self.u_friction,
                ", z_0 = ",
                self.z_0,
            )

        elif self.profile_type == "ud_exponential":
            self.mean_profile = self.CreateUdExpMeanProfile()
            Logger.PrintInfo(
                "\n## Sponge Layer Condition Summary ##",
                "\nImposeSpongeLayerProcess: ",
                "Adopting mean exponential velocity profile with: ",
                "z_ref = ",
                self.z_ref,
                ", u_mean = ",
                self.u_mean,
                ", alpha = ",
                self.alpha,
            )

        else:
            raise Exception(
                "Value ",
                self.profile_type,
                " not possible for profile_type.",
                "Possible values are : ud_exponential, logarithmic",
            )

        # Print Sponge layer parameters to screen :

        if self.print_to_screen:

            result_msg = (
                "Sponge Layer Parameters: "
                "\n"
                + "Damping Coefficent: "
                + str(self.damping_coefficent)
                + "\n"
                + "Sponge Layer Length: "
                + str(self.d_)
                + "\n"
            )
            self._PrintToScreen(result_msg)

    #########################################

    def _PrintToScreen(self, result_msg):
        Logger.PrintInfo("\nImposeSpongeLayerProcess ", result_msg)

    #########################################

    def CreateLogMeanProfile(self, key=lambda node: node.Z, von_karman_const=0.41):
        z_0 = self.z_0
        u_friction = self.u_friction
        return LogMeanProfile(u_friction, z_0)

    #########################################

    def CreateUdExpMeanProfile(self, key=lambda node: node.Z):
        u_mean = self.u_mean
        z_ref = self.z_ref
        alpha = self.alpha
        return UdExpMeanProfile(u_mean, z_ref, alpha)

    #########################################

    def ExecuteInitialize(self):
        for node in self.sponge_nodes:
            node.Fix(BODY_FORCE_X)
            node.Fix(BODY_FORCE_Y)
            node.Fix(BODY_FORCE_Z)

    #########################################

    def ExecuteFinalize(self):
        if self.model_part.GetCommunicator().MyPID() == 0:
            pass

    #########################################

    def ExecuteInitializeSolutionStep(self):

        for node in self.Sponge_layer_nodes:

            # Obtain mean velocity as per prescribed velocity profile (streanwise component)

            vel = self.mean_profile.mean_wind_speed(node)

            # Scaled damping coefficent

            self.F_d = (node.X - self.x_start) * self.F_k

            # Assign damping forces (Fx , Fy, Fz)

            node.SetSolutionStepValue(
                BODY_FORCE_X,
                0,
                -self.F_d * (node.GetSolutionStepValue(VELOCITY_X) - vel),
            )

            node.SetSolutionStepValue(
                BODY_FORCE_Y, 0, -self.F_d * (node.GetSolutionStepValue(VELOCITY_Y))
            )

            node.SetSolutionStepValue(
                BODY_FORCE_Z, 0, -self.F_d * (node.GetSolutionStepValue(VELOCITY_Z))
            )

    #########################################

    def Check(self):
        pass

    #########################################

    def ExecuteBeforeSolutionLoop(self):
        pass

    #########################################

    def ExecuteFinalizeSolutionStep(self):
        pass

    #########################################

    def ExecuteBeforeOutputStep(self):
        pass

    #########################################

    def ExecuteAfterOutputStep(self):
        pass
