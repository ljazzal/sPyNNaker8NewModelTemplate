# import spynnaker8 and plotting stuff
# import spynnaker8 as p
import pyNN.spiNNaker as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# import models
from python_models8.neuron.builds.my_full_neuron import MyFullNeuron

# Set the run time of the execution
run_time = 1000

# Set the time step of the simulation in milliseconds
time_step = 1.0

# Set the number of neurons to simulate
n_neurons = 1

# Set the i_offset current
i_offset = 0.0

# Set the weight of input spikes
weight = 2.0

# Set the times at which to input a spike
spike_times = range(0, run_time, 100)

p.setup(time_step)

spikeArray = {"spike_times": spike_times}
input_pop = p.Population(
    n_neurons, p.SpikeSourceArray(**spikeArray), label="input")

pulse = p.DCSource(amplitude=10.0, start=100.0, stop=600.0)

my_full_neuron_pop = p.Population(
    n_neurons, MyFullNeuron(), label="my_full_neuron_pop")
p.Projection(
    input_pop, my_full_neuron_pop,
    p.OneToOneConnector(), receptor_type='excitatory',
    synapse_type=p.StaticSynapse(weight=weight))

my_full_neuron_pop.inject(pulse)

my_full_neuron_pop.record(['v', 'ext_input'])

p.run(run_time)

# get v for each example
data = my_full_neuron_pop.get_data()
v_my_full_neuron_pop = my_full_neuron_pop.get_data('v')
I_ext_my_full_neuron_pop = my_full_neuron_pop.get_data('ext_input')

Figure(
    # membrane potentials for each example
    Panel(v_my_full_neuron_pop.segments[0].filter(name='v')[0],
          xlabel="Time (ms)",
          ylabel="Membrane potential (mV)",
          data_labels=[my_full_neuron_pop.label],
          yticks=True, xlim=(0, run_time), xticks=True),
    Panel(I_ext_my_full_neuron_pop.segments[0].filter(name='ext_input')[0],
          xlabel="Time (ms)",
          ylabel="External current",
          data_labels=[my_full_neuron_pop.label],
          yticks=True, xlim=(0, run_time), xticks=True),
    title="Simple my model examples",
    annotations="Simulated with {}".format(p.name())
)
plt.show()

p.end()
