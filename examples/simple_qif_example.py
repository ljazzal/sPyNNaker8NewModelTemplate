# Test simple_qif model on SpiNNaker
import pyNN.spiNNaker as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np

# import models
from python_models8.neuron.builds.simple_qif import SimpleQif


# Set the run time of the execution
run_time = 20

# Set the time step of the simulation in milliseconds
# time_step = 1.0
time_step = 1.0

# Set the number of neurons to simulate
n_neurons = 10000

# Set the i_offset current
i_offset = 5.0

# Set the times at which to input a spike
spike_times = range(0, run_time, 100)

spikeArray = {"spike_times": spike_times}

sim.setup(timestep=0.001)

input_pop = sim.Population(
    n_neurons, sim.SpikeSourceArray(**spikeArray), label="input")

qif_pop = sim.Population(
    # n_neurons, sim.Izhikevich(i_offset=i_offset),
    n_neurons, SimpleQif(i_offset=np.random.rand(n_neurons)),
    label="qif_pop")

poisson_input = sim.Population(n_neurons, sim.SpikeSourcePoisson(rate=1000.0))
sim.Projection(poisson_input, qif_pop, sim.FixedProbabilityConnector(0.1), sim.StaticSynapse(weight=0.11), receptor_type="excitatory")


# sim.Projection(
#     input_pop, qif_pop, sim.AllToAllConnector(), receptor_type='excitatory',
#     synapse_type=sim.StaticSynapse(weight=2.0))
sim.Projection(
    input_pop, qif_pop, sim.FixedProbabilityConnector(0.1), receptor_type='excitatory',
    synapse_type=sim.StaticSynapse(weight=2.0))

# qif_pop.record(['v', 'spikes', 'gsyn_exc'])
qif_pop.record(['spikes'])

sim.run(run_time)

# get v for each example
# v_qif_pop = qif_pop.get_data('v').segments[0].filter(name='v')[0]
# I_qif_pop = qif_pop.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]
spikes = qif_pop.get_data('spikes').segments[0].spiketrains

sim.end()

Figure(
    # membrane potentials for each example
    # Panel(v_qif_pop,
    #     ylabel="Excitatory current",
    #     yticks=True, xticks=True, xlim=(0, run_time)),
    # Panel(I_qif_pop,
    #       ylabel="Membrane potential (mV)",
    #       yticks=True, xticks=True, xlim=(0, run_time)),
    Panel(spikes,
          ylabel="Spike",
          yticks=True, xticks=True, xlim=(0, run_time)),
    title="Simple QIF model example"
)
plt.show()
