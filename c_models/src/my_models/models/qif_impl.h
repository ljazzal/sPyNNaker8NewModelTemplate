/*
 * Copyright (c) 2017-2019 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

//! \file
//! \brief Quadratic integrate-and-fire (QIF) neuron type
#ifndef _QIF_IMPL_H_
#define _QIF_IMPL_H_

#include "neuron_model.h"

//! The state variables of an QIF model neuron
typedef struct neuron_t {
    // nominally 'fixed' parameters
    REAL C;

    // Variable-state parameters
    REAL V;

    //! offset current [nA]
    REAL I_offset;

    //! current timestep - simple correction for threshold
    REAL this_h;
} neuron_t;

//! Global neuron parameters for QIF model neuron
typedef struct global_neuron_params_t {
    REAL machine_timestep_ms;
} global_neuron_params_t;

extern const global_neuron_params_t *global_params;

/*! \brief For linear membrane voltages, 1.5 is the correct value. However
 * with actual membrane voltage behaviour and tested over an wide range of
 * use cases 1.85 gives slightly better spike timings.
 */
static const REAL SIMPLE_TQ_OFFSET = REAL_CONST(1.85);

/////////////////////////////////////////////////////////////
#if 0
// definition for Izhikevich neuron
static inline void neuron_ode(
        REAL t, REAL stateVar[], REAL dstateVar_dt[],
        neuron_t *neuron, REAL input_this_timestep) {
    REAL V_now = stateVar[1];
    REAL U_now = stateVar[2];
    log_debug(" sv1 %9.4k  V %9.4k --- sv2 %9.4k  U %9.4k\n", stateVar[1],
            neuron->V, stateVar[2], neuron->U);

    // Update V
    dstateVar_dt[1] =
            REAL_CONST(140.0)
            + (REAL_CONST(5.0) + REAL_CONST(0.0400) * V_now) * V_now - U_now
            + input_this_timestep;

    // Update U
    dstateVar_dt[2] = neuron->A * (neuron->B * V_now - U_now);
}
#endif

/*!
 * \brief Midpoint is best balance between speed and accuracy so far.
 * \details From ODE solver comparison work, paper shows that Trapezoid version
 *      gives better accuracy at small speed cost
 * \param[in] h: threshold
 * \param[in,out] neuron: The model being updated
 * \param[in] input_this_timestep: the input
 */
static inline void rk2_kernel_midpoint(
        REAL h, neuron_t *neuron, REAL input_this_timestep) {
    // to match Mathematica names
    REAL lastV1 = neuron->V;

    REAL pre_alph = input_this_timestep;
    REAL alpha = pre_alph + lastV1 * lastV1;
    REAL eta = lastV1 + REAL_HALF(h * alpha);

    neuron->V += h * (pre_alph + eta * eta);
}


static state_t neuron_model_state_update(
        uint16_t num_excitatory_inputs, const input_t *exc_input,
    uint16_t num_inhibitory_inputs, const input_t *inh_input,
    input_t external_bias, neuron_t *restrict neuron) {
    REAL total_exc = 0;
    REAL total_inh = 0;

    for (int i =0; i<num_excitatory_inputs; i++) {
        total_exc += exc_input[i];
    }
    for (int i =0; i<num_inhibitory_inputs; i++) {
        total_inh += inh_input[i];
    }

    input_t input_this_timestep = total_exc - total_inh
            + external_bias + neuron->I_offset;

    // the best AR update so far
    rk2_kernel_midpoint(neuron->this_h, neuron, input_this_timestep);
    neuron->this_h = global_params->machine_timestep_ms;

    return neuron->V;
}

static void neuron_model_has_spiked(neuron_t *restrict neuron) {
    // reset membrane voltage
    neuron->V = neuron->C;

    // simple threshold correction - next timestep (only) gets a bump
    neuron->this_h = global_params->machine_timestep_ms * SIMPLE_TQ_OFFSET;
}

static state_t neuron_model_get_membrane_voltage(const neuron_t *neuron) {
    return neuron->V;
}

#endif   // _QIF_IMPL_H_
