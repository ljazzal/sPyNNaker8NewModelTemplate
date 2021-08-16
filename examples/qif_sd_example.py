# Test qif_sd model on SpiNNaker
import pyNN.spiNNaker as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np

# import models
from python_models8.neuron.builds.qif_sd import QifSd


# Set the run time of the execution
run_time = 20

# Set the time step of the simulation in milliseconds
# time_step = 1.0
time_step = 0.001

# Set the number of neurons to simulate
n_neurons = 5

# Set the i_offset current
# i_offset = 5.0
i_offset = np.random.randn(n_neurons)

# Set the times at which to input a spike
spike_times = range(0, run_time, 100)

spikeArray = {"spike_times": spike_times}

sim.setup(timestep=time_step)

# input_pop = sim.Population(
#     n_neurons, sim.SpikeSourceArray(**spikeArray), label="input")
input_pop = sim.Population(
    n_neurons, sim.SpikeSourcePoisson(rate=50), label="input")
input_pop1 = sim.Population(
    n_neurons, sim.SpikeSourcePoisson(rate=100), label="input")

qif_pop = sim.Population(
    # n_neurons, sim.Izhikevich(i_offset=i_offset),
    n_neurons, QifSd(i_offset=np.random.rand(n_neurons)),
    # n_neurons, sim.IF_curr_alpha(),
    label="qif_pop")

sim.Projection(
    input_pop1, qif_pop, sim.FixedProbabilityConnector(p_connect=0.5), receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=5.0))

sim.Projection(
    input_pop, qif_pop, sim.FixedProbabilityConnector(p_connect=0.8), receptor_type='inhibitory',
    synapse_type=sim.StaticSynapse(weight=-1.0))

# qif_pop.record(['v', 'spikes', 'gsyn_exc'])
qif_pop.record(['v', 'spikes', 'gsyn_exc', 'gsyn_inh'])

sim.run(run_time)

# get v for each example
data = qif_pop.get_data()
v_qif_pop = qif_pop.get_data('v').segments[0].filter(name='v')[0]
I_qif_exc = qif_pop.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]
I_qif_inh = qif_pop.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]
spikes = qif_pop.get_data('spikes').segments[0].spiketrains

sim.end()

Figure(
    # membrane potentials for each example
    Panel(v_qif_pop,
        ylabel="Membrane potential",
        yticks=True, xticks=True, xlim=(0, run_time)),
    Panel(I_qif_exc,
        ylabel="Excitatory current",
        yticks=True, xticks=True, xlim=(0, run_time)),
    Panel(I_qif_inh,
          ylabel="Inhibitory current",
          yticks=True, xticks=True, xlim=(0, run_time)),
    Panel(spikes,
          ylabel="Spike",
          yticks=True, xticks=True, xlim=(0, run_time)),
    title="Simple QIF model example"
)
plt.show()
