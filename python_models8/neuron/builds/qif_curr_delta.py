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

from spynnaker.pyNN.models.neuron.synapse_types import SynapseTypeDelta
from spynnaker.pyNN.models.neuron.input_types import InputTypeCurrent
from spynnaker.pyNN.models.neuron.threshold_types import ThresholdTypeStatic
from spynnaker.pyNN.models.neuron import AbstractPyNNNeuronModelStandard
from spynnaker.pyNN.models.defaults import default_initial_values

from python_models8.neuron.neuron_models.neuron_model_quadratic_integrate_and_fire import (
    NeuronModelQuadraticIntegrateAndFire)

_IZK_THRESHOLD = 100.0


class QIFCurrDelta(AbstractPyNNNeuronModelStandard):
    """ Izhikevich neuron model with current inputs.

    :param a: :math:`a`
    :type a: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    :param b: :math:`b`
    :type b: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    :param c: :math:`c`
    :type c: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    :param d: :math:`d`
    :type d: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    :param i_offset: :math:`I_{offset}`
    :type i_offset: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    :param u: :math:`u_{init} = \\delta V_{init}`
    :type u: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    :param v: :math:`v_{init} = V_{init}`
    :type v: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    :param tau_syn_E: :math:`\\tau^{syn}_e`
    :type tau_syn_E: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    :param tau_syn_I: :math:`\\tau^{syn}_i`
    :type tau_syn_I: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    :param isyn_exc: :math:`I^{syn}_e`
    :type isyn_exc: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    :param isyn_inh: :math:`I^{syn}_i`
    :type isyn_inh: float, iterable(float), ~pyNN.random.RandomDistribution
        or (mapping) function
    """

    # noinspection PyPep8Naming
    @default_initial_values({"v", "isyn_exc", "isyn_inh"})
    def __init__(self, c=-100.0, i_offset=0.0, v=-100.0, tau_refrac=0.002,
        isyn_exc=0.0, isyn_inh=0.0):
        # pylint: disable=too-many-arguments, too-many-locals
        neuron_model = NeuronModelQuadraticIntegrateAndFire(c, v, i_offset, tau_refrac)
        synapse_type = SynapseTypeDelta(isyn_exc, isyn_inh)
        input_type = InputTypeCurrent()
        threshold_type = ThresholdTypeStatic(_IZK_THRESHOLD)

        super().__init__(
            model_name="QIFCurrDelta", binary="QIF_curr_delta.aplx",
            neuron_model=neuron_model, input_type=input_type,
            synapse_type=synapse_type, threshold_type=threshold_type)
