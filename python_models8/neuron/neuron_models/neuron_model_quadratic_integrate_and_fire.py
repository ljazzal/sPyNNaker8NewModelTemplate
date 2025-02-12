# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy
from spinn_utilities.overrides import overrides
from data_specification.enums import DataType
from spinn_front_end_common.utilities.constants import (
    MICRO_TO_MILLISECOND_CONVERSION)
from spynnaker.pyNN.models.neuron.neuron_models import AbstractNeuronModel
from spynnaker.pyNN.models.neuron.implementations import (
    AbstractStandardNeuronComponent)

# TODO: replace 'C' with 'V_RESET'
C = 'c'
V = 'v'
I_OFFSET = 'i_offset'
TAU_REFRAC = 'tau_refrac'
COUNT_REFRAC = 'count_refrac'

UNITS = {
    C: "mV",
    V: "mV",
    I_OFFSET: "nA",
    TAU_REFRAC: "ms"
}


class NeuronModelQuadraticIntegrateAndFire(AbstractNeuronModel):
    """ QIF model (simplified Izhikevich model)
    """
    __slots__ = [
        "__c", "__v_init", "__i_offset", "__tau_refrac"
    ]

    def __init__(self, c, v_init, i_offset, tau_refrac):
        """
        :param c: :math:`c`
        :type c: float, iterable(float), ~pyNN.random.RandomDistribution or
            (mapping) function
        :param v_init: :math:`v_{init}`
        :type v_init:
            float, iterable(float), ~pyNN.random.RandomDistribution or
            (mapping) function
        :param i_offset: :math:`I_{offset}`
        :type i_offset:
            float, iterable(float), ~pyNN.random.RandomDistribution or
            (mapping) function
        """
        super().__init__(
            [DataType.S1615,   # c
             DataType.S1615,   # v
             DataType.S1615,   # i_offset
             DataType.INT32,   # count_refrac
             DataType.INT32,   # tau_refrac
             DataType.S1615],  # this_h (= machine_time_step)
            [DataType.S1615])  # machine_time_step
        self.__c = c
        self.__i_offset = i_offset
        self.__v_init = v_init
        self.__tau_refrac = tau_refrac

    @overrides(AbstractStandardNeuronComponent.get_n_cpu_cycles)
    def get_n_cpu_cycles(self, n_neurons):
        # A bit of a guess
        return 200 * n_neurons

    @overrides(AbstractStandardNeuronComponent.add_parameters)
    def add_parameters(self, parameters):
        parameters[C] = self.__c
        parameters[I_OFFSET] = self.__i_offset
        parameters[TAU_REFRAC] = self.__tau_refrac

    @overrides(AbstractStandardNeuronComponent.add_state_variables)
    def add_state_variables(self, state_variables):
        state_variables[V] = self.__v_init
        state_variables[COUNT_REFRAC] = 0

    @overrides(AbstractStandardNeuronComponent.get_units)
    def get_units(self, variable):
        return UNITS[variable]

    @overrides(AbstractStandardNeuronComponent.has_variable)
    def has_variable(self, variable):
        return variable in UNITS

    @overrides(AbstractNeuronModel.get_global_values)
    def get_global_values(self, ts):
        # pylint: disable=arguments-differ
        return [float(ts) / MICRO_TO_MILLISECOND_CONVERSION]

    @overrides(AbstractStandardNeuronComponent.get_values)
    def get_values(self, parameters, state_variables, vertex_slice, ts):
        """
        :param ts: machine time step
        """
        # pylint: disable=arguments-differ

        # Add the rest of the data
        return [
            parameters[C],
            state_variables[V], parameters[I_OFFSET],
            state_variables[COUNT_REFRAC], parameters[TAU_REFRAC].apply_operation(
                operation=lambda x: int(numpy.ceil(x / (ts / 1000.0)))),
            float(ts) / MICRO_TO_MILLISECOND_CONVERSION
        ]

    @overrides(AbstractStandardNeuronComponent.update_values)
    def update_values(self, values, parameters, state_variables):

        # Decode the values
        _c, v, _i_offset, count_refrac, _tau_refrac, _this_h = values

        # Copy the changed data only
        state_variables[V] = v
        state_variables[COUNT_REFRAC] = count_refrac

    @property
    def c(self):
        """ Settable model parameter: :math:`c`

        :rtype: float
        """
        return self.__c

    @property
    def i_offset(self):
        """ Settable model parameter: :math:`I_{offset}`

        :rtype: float
        """
        return self.__i_offset

    @property
    def v_init(self):
        """ Settable model parameter: :math:`v_{init}`

        :rtype: float
        """
        return self.__v_init

    @property
    def tau_refrac(self):
        r""" Settable model parameter: :math:`\tau_{refrac}`

        :rtype: float
        """
        return self.__tau_refrac
