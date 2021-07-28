"""
Convective outflow condition for FluidDynamics in KratosMultiphysics

"""

######################################################################
# Chair of Structural Analysis - TUM (Statik)
# Author: Ammar Khallouf
# Date: 04.05.21
# Version: 1.0
######################################################################
""" 
*** Description ***

In this process the velocity and pressure at the outlet boundary are interpolated from the interior of the fluid domain.

The convective conditon follows a one dimensional wave equation with phase speed "c" estimated from outlet neighbouring nodes

It's combined with a relaxation term to avoid reference pressure drifting. The relaxation term represent the time required by the flow waves

to traverl from inlet to outlet and hence based on domain length and inlet mean flow velocity.

"""
######################################################################
"""
*** References ***

Khallouf, A: Developments for Improved FEM-Based LES Simulations in Computational Wind Engineering (Master's Thesis)(2021)

Hirt, CW: Addition of Wave Transmitting boundary conditions to the FLOW-3D program. In: Flow Science Inc., Technical Note, FSI-99-TN49 (1999)

"""
######################################################################

"""
Example of usage:

Required Input:

1. Volume model part for the fluid domain.

2. Surface model part for the outlet boundary.

3. Far-field reference pressure: p_inf (recommended to be zero static pressure -> p_inf= 0.0 ).

4. Far-field reference length : L_inf (recommnended to be domain streamwise length)

4. Mean reference flow velocity : U_ref (recommnended to be the average velocity at the inlet boundary)

{
    "python_module" : "convective_outlet_process",
    "Parameters"    : {
        "fluid_model_part_name" :  "FluidModelPart.ModelPartName",
        "outlet_model_part_name" : "FluidModelPart.ModelPartName",
        "print_to_screen": false,
        "P_inf"  :0.00,
        "L_inf"  :1800,
        "U_ref"      : 40.0
    }
}

"""

######################################################################

__all__ = ["Factory", "ImposeConvectiveOutletProcess"]

import os

from collections import namedtuple, Mapping
from math import isclose, floor, ceil, log

from KratosMultiphysics import VELOCITY_X, VELOCITY_Y, VELOCITY_Z, PRESSURE
from KratosMultiphysics import BODY_FORCE_X, BODY_FORCE_Y, BODY_FORCE_Z
from KratosMultiphysics import TIME, DELTA_TIME, NodesArray
from KratosMultiphysics import (
    Model,
    Logger,
    DataCommunicator,
    FindGlobalNodalNeighboursProcess,
)

# other imports
from KratosMultiphysics.time_based_ascii_file_writer_utility import (
    TimeBasedAsciiFileWriterUtility,
)
from KratosMultiphysics import Parameters as KM_Parameters
from KratosMultiphysics import IS_RESTARTED as KM_IS_RESTARTED

kratos_comm = DataCommunicator.GetDefault()

######################################################################


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
    return ImposeConvectiveOutletProcess(Model, Parameters(settings["Parameters"]))


# -------------------------------Convective Outlet Utility-----------------------------------


class ImposeConvectiveOutletProcess:
    @property
    def outlet_nodes(self):

        # Return the nodes for the outlet boundary

        return self.outlet_model_part.Nodes

    #########################################

    def __init__(self, Model, settings):

        # Read input settings form .json file

        for name, value in settings.items():

            setattr(self, name, value)

        # Read volume model part name

        self.fluid_model_part = Model[self.fluid_model_part_name]

        # Read outlet model part name

        self.outlet_model_part = Model[self.outlet_model_part_name]

        # ------------------------------------------------------------------

        # Nodes 1st level neighbour process execution

        self.neighbour_search = FindGlobalNodalNeighboursProcess(
            kratos_comm, self.fluid_model_part
        )
        self.neighbour_search.Execute()

        # Get Nodes ID's direct neighbour to the outlet (as dictionary)

        self.neighbour_1_ids = self.neighbour_search.GetNeighbourIds(
            self.outlet_model_part.Nodes
        )

        # Parse neighbours dictonary into lists

        # ---> contains outlet node ID's
        self.neighbour_1_dict_keys = list(self.neighbour_1_ids.keys())

        # --> contains 1st level neighbour node ID's

        self.neighbour_1_dict_values = list(self.neighbour_1_ids.values())

        # Flatten the list containning the 1st level neighbour node ID's

        self.neighbour_1_dict_values = [
            val for sublist in self.neighbour_1_dict_values for val in sublist
        ]

        # Remove duplicates

        self.neighbour_1_dict_values = list(set(self.neighbour_1_dict_values))

        self.neighbour_1_nodes_ids = list(
            set(self.neighbour_1_dict_values) ^ set(self.neighbour_1_dict_keys)
        )

        # sort nodes id's in asccending order

        self.neighbour_1_nodes_ids.sort()

        self.Neighbour_1_Nodes = NodesArray()

        # Extract the 1st level neighbour node ID's to the outlet model part

        for id_num in self.neighbour_1_nodes_ids:
            self.Neighbour_1_Nodes.append(self.fluid_model_part.Nodes[id_num])

        # ------------------------------------------------------------------

        # Nodes 2nd level neighbour process execution

        self.neighbour_2_ids = self.neighbour_search.GetNeighbourIds(
            self.Neighbour_1_Nodes
        )

        # Parse neighbours dictonary into lists

        # ---> contains 1st neighbours node ID's
        self.neighbour_2_dict_keys = list(self.neighbour_2_ids.keys())

        # --> contains 2nd neighbours node ID's

        self.neighbour_2_dict_values = list(self.neighbour_2_ids.values())

        # Flatten the list containning the neighbour node ID's

        self.neighbour_2_dict_values = [
            val for sublist in self.neighbour_2_dict_values for val in sublist
        ]

        # Remove duplicates with respect to outlet nodes and 1st level neighbours

        self.neighbour_2_dict_values = list(set(self.neighbour_2_dict_values))

        self.neighbour_2_nodes_ids = list(
            set(self.neighbour_2_dict_values) ^ set(self.neighbour_2_dict_keys)
        )

        self.neighbour_2_nodes_ids = list(
            set(self.neighbour_2_nodes_ids) ^ set(self.neighbour_1_dict_keys)
        )

        # sort nodes id's in asccending order

        self.neighbour_2_nodes_ids.sort()

        self.Neighbour_2_Nodes = NodesArray()

        # Extract the 2nd level neighbour node ID's to the outlet model part

        for id_num in self.neighbour_2_nodes_ids:
            self.Neighbour_2_Nodes.append(self.fluid_model_part.Nodes[id_num])

        # ------------------------------------------------------------------

        # Handle exceptions for inappropraite far field length value (L_inf)

        if self.L_inf <= 0.0:
            raise Exception(
                "Invalid value for L_inf ",
                " ## L_inf should be strictly > 0.0 ##",
            )

        # Print Convective Outlet parameters to screen :

        if self.print_to_screen:

            result_msg = (
                "### Convective Condition Parameters ###"
                "\n"
                + "Far-field pressure P_inf: "
                + str(self.P_inf)
                + "\n"
                + "Far-field length L_inf: "
                + str(self.L_inf)
                + "\n"
                + "Reference flow velocity U_ref: "
                + str(self.U_ref)
                + "\n"
            )
            self._PrintToScreen(result_msg)

    #########################################

    def _PrintToScreen(self, result_msg):
        Logger.PrintInfo("\nImposeConvectiveOutletProcess ", result_msg)

    #########################################

    def ExecuteInitialize(self):

        # Define Lists to store boundary values

        self.predicted_pressure = []

        self.predicted_velocity_x = []

        self.predicted_velocity_y = []

        self.predicted_velocity_z = []

        # Fix DOF's and store initial values

        for node in self.outlet_nodes:
            node.Fix(VELOCITY_X)
            node.Fix(VELOCITY_Y)
            node.Fix(VELOCITY_Z)
            node.Fix(PRESSURE)

            self.predicted_pressure.append(node.GetSolutionStepValue(PRESSURE))

            self.predicted_velocity_x.append(node.GetSolutionStepValue(VELOCITY_X))

            self.predicted_velocity_y.append(node.GetSolutionStepValue(VELOCITY_Y))

            self.predicted_velocity_z.append(node.GetSolutionStepValue(VELOCITY_Z))

    #########################################

    def ExecuteInitializeSolutionStep(self):

        # Define Lists to store current neighbours values (n)

        self.q_i1 = []  # Pressure at direct neighbours
        self.q_i2 = []  # Pressure at neighbours of neighbours

        self.v_xi1 = []  # Vx at direct neighbours
        self.v_xi2 = []  # Vx at neighbours of neighbours

        self.v_yi1 = []  # Vy at direct neighbours
        self.v_yi2 = []  # Vy at neighbours of neighbours

        self.v_zi1 = []  # Vz at direct neighbours
        self.v_zi2 = []  # Vz at neighbours of neighbours

        ######################################################################

        # 1. ->  Loop over 2 levels neighbouring nodes of the outlet and get cuurent values

        ######################################################################

        for node_l1, node_l2 in zip(self.Neighbour_1_Nodes, self.Neighbour_2_Nodes):

            self.q_i1.append(node_l1.GetSolutionStepValue(PRESSURE))

            self.v_xi1.append(node_l1.GetSolutionStepValue(VELOCITY_X))

            self.v_yi1.append(node_l1.GetSolutionStepValue(VELOCITY_Y))

            self.v_zi1.append(node_l1.GetSolutionStepValue(VELOCITY_Z))

            self.q_i2.append(node_l2.GetSolutionStepValue(PRESSURE))

            self.v_xi2.append(node_l2.GetSolutionStepValue(VELOCITY_X))

            self.v_yi2.append(node_l2.GetSolutionStepValue(VELOCITY_Y))

            self.v_zi2.append(node_l2.GetSolutionStepValue(VELOCITY_Z))

        ######################################################################

        # 2. ->   Assign interpolated vales to the boundary

        ######################################################################

        i = 0

        for node in self.outlet_nodes:

            node.SetSolutionStepValue(PRESSURE, 0, self.predicted_pressure[i])

            node.SetSolutionStepValue(VELOCITY_X, 0, self.predicted_velocity_x[i])

            node.SetSolutionStepValue(VELOCITY_Y, 0, self.predicted_velocity_y[i])

            node.SetSolutionStepValue(VELOCITY_Z, 0, self.predicted_velocity_z[i])

            i += 1

    #########################################

    def ExecuteFinalizeSolutionStep(self):

        self.q_i1_nxt = []  # Pressure at direct neighbours
        self.q_i2_nxt = []  # Pressure at neighbours of neighbours

        self.v_xi1_nxt = []  # Vx at direct neighbours
        self.v_xi2_nxt = []  # Vx at neighbours of neighbours

        self.v_yi1_nxt = []  # Vy at direct neighbours
        self.v_yi2_nxt = []  # Vy at neighbours of neighbours

        self.v_zi1_nxt = []  # Vz at direct neighbours
        self.v_zi2_nxt = []  # Vz at neighbours of neighbours

        self.c_p = 0.0  # Non-dimensional wave speed for pressure waves.
        self.c_vx = 0.0  # Non-dimensional wave speed for Vx waves.
        self.c_vy = 0.0  # Non-dimensional wave speed for Vy waves.
        self.c_vz = 0.0  # Non-dimensional wave speed for Vz waves.

        ######################################################################

        # 1. ->  Loop over 2 levels neighbouring nodes of the outlet and get new values

        ######################################################################

        for node_l1, node_l2 in zip(self.Neighbour_1_Nodes, self.Neighbour_2_Nodes):

            self.q_i1_nxt.append(node_l1.GetSolutionStepValue(PRESSURE))

            self.v_xi1_nxt.append(node_l1.GetSolutionStepValue(VELOCITY_X))

            self.v_yi1_nxt.append(node_l1.GetSolutionStepValue(VELOCITY_Y))

            self.v_zi1_nxt.append(node_l1.GetSolutionStepValue(VELOCITY_Z))

            self.q_i2_nxt.append(node_l2.GetSolutionStepValue(PRESSURE))

            self.v_xi2_nxt.append(node_l2.GetSolutionStepValue(VELOCITY_X))

            self.v_yi2_nxt.append(node_l2.GetSolutionStepValue(VELOCITY_Y))

            self.v_zi2_nxt.append(node_l2.GetSolutionStepValue(VELOCITY_Z))

        ######################################################################

        # 2. ->  Calculate New Boundary Values for Next Cycle of Computation

        ######################################################################

        # TO IMPROVE: Create one to one mapping between outlet nodes and neighbour nodes for the case (neighbour nodes count > outlet nodes count)

        # Calculate predicted velocity and pressure for next time step

        # print(len(self.predicted_pressure), len(self.q_i1_nxt), len(self.q_i2_nxt))

        for j in range(len(self.predicted_pressure)):

            ################################ For Pressure ####################################

            # Claculate c_p using neighbour values

            self.nominator = (
                self.q_i1_nxt[j] - self.q_i2[j] - self.q_i1[j] + self.q_i2_nxt[j]
            )

            self.dominator = (
                -self.q_i1[j] + self.q_i2_nxt[j] - self.q_i1_nxt[j] + self.q_i2[j]
            )

            if abs(self.dominator) > 1.0e-6:

                # Limit values between 0 and 1

                self.c_p = max(min(1, (self.nominator / self.dominator)), 0.0)

                # Calculate new boundary value for pressure -> to be assigned at next step

                self.predicted_pressure[j] = self.q_i1[j] + (
                    (1.0 - self.c_p) / (1.0 + self.c_p)
                ) * (self.predicted_pressure[j] - self.q_i1_nxt[j])

                # Relax the pressure value based on (K = U_ref/L_inf) and far field value (P_inf)

                self.predicted_pressure[j] += (self.U_ref / self.L_inf) * (
                    self.P_inf - self.predicted_pressure[j]
                )

            # print((1.0-self.c_p)/(1.0+self.c_p))

            ################################ For Vx ####################################

            # Claculate c_vx using neighbour values

            self.nominator = (
                self.v_xi1_nxt[j] - self.v_xi2[j] - self.v_xi1[j] + self.v_xi2_nxt[j]
            )

            self.dominator = (
                -self.v_xi1[j] + self.v_xi2_nxt[j] - self.v_xi1_nxt[j] + self.v_xi2[j]
            )

            if abs(self.dominator) > 1.0e-6:

                # Limit values between 0 and 1

                self.c_vx = max(min(1, self.nominator / self.dominator), 0.0)

                # Calculate new boundary value for Vx -> to be assigned at next step

                self.predicted_velocity_x[j] = self.v_xi1[j] + (
                    (1.0 - self.c_vx) / (1.0 + self.c_vx)
                ) * (self.predicted_velocity_x[j] - self.v_xi1_nxt[j])

            # print((1.0-self.c_vx)/(1.0+self.c_vx))

            ################################ For Vy ####################################

            # Claculate c_vy using neighbour values

            self.nominator = (
                self.v_yi1_nxt[j] - self.v_yi2[j] - self.v_yi1[j] + self.v_yi2_nxt[j]
            )

            self.dominator = (
                -self.v_yi1[j] + self.v_yi2_nxt[j] - self.v_yi1_nxt[j] + self.v_yi2[j]
            )

            if abs(self.dominator) > 1.0e-6:

                # Limit values between 0 and 1

                self.c_vy = max(min(1, self.nominator / self.dominator), 0.0)

                # Calculate new boundary value for Vy -> to be assigned at next step

                self.predicted_velocity_y[j] = self.v_yi1[j] + (
                    (1.0 - self.c_vy) / (1.0 + self.c_vy)
                ) * (self.predicted_velocity_y[j] - self.v_yi1_nxt[j])

            # print((1.0-self.c_vy)/(1.0+self.c_vy))

            ################################ For Vz ####################################

            # Claculate c_vz using neighbour values

            self.nominator = (
                self.v_zi1_nxt[j] - self.v_zi2[j] - self.v_zi1[j] + self.v_zi2_nxt[j]
            )

            self.dominator = (
                -self.v_zi1[j] + self.v_zi2_nxt[j] - self.v_zi1_nxt[j] + self.v_zi2[j]
            )

            if abs(self.dominator) > 1.0e-6:

                # Limit values between 0 and 1

                self.c_vz = max(min(1, self.nominator / self.dominator), 0.0)

                # Calculate new boundary value for Vz -> to be assigned at next step

                self.predicted_velocity_z[j] = self.v_zi1[j] + (
                    (1.0 - self.c_vz) / (1.0 + self.c_vz)
                ) * (self.predicted_velocity_z[j] - self.v_zi1_nxt[j])

            # print((1.0-self.c_vz)/(1.0+self.c_vz))

    #########################################

    def ExecuteFinalize(self):
        if self.model_part.GetCommunicator().MyPID() == 0:
            pass

    #########################################

    def Check(self):
        pass

    #########################################

    def ExecuteBeforeSolutionLoop(self):
        pass

    #########################################

    def ExecuteBeforeOutputStep(self):
        pass

    #########################################

    def ExecuteAfterOutputStep(self):
        pass
