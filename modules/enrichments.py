"""
"""

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

class Growth_tube:
    """Growth tube class to drop species in and simulate enrichments.

    Parameters
    ----------
    x_i0 : np.array
        Array of initial species values in the initial population (before dilution).
    r_i : np.array
        Array of the species growth rates.
    dil : int
        Dilution in part-to-whole format (e.g. 1:100 dilution; d = 100).

    """

    def __init__(self, x_i0, r_i, dil):
        """Short summary.

        Parameters
        ----------
        x_i0 : np.array
            Array of initial species values in the initial population (before dilution).
        r_i : np.array
            Array of the species growth rates.
        dil : int
            Dilution in part-to-whole format (e.g. 1:100 dilution; d = 100).
        x_i0_dil: np.array
            Initial species values after uniform dilution. (May want to play with
            sampling creating different distributions of diluted species at some point).
        x_t: np.array
            Matrix holding species values at each timepoint.
        t: np.array
            Timepoint values.
        enrichs: np.array
            Enrichment values for each species after simulation.
        """

        assert (np.shape(x_i0) == np.shape(r_i)), "Initial values and rates must be the same size!"

        self.x_i0 = x_i0
        assert (np.isclose(np.sum(x_i0), 1, rtol = 1e-5)), "Initial fractions must add up to 1!"
        self.r_i = r_i
        self.d = dil

        # Calculate the diluted culture
        #   NOTE: May want to do fancier things here to look at stochasticity when diluting.
        self.x_i0_dil = x_i0 / self.d

        # Empty attributes to be filled with the simulation
        self.x_t = np.empty(0)
        self.t = np.empty(0)
        self.enrichs = np.empty(0)

    def __ode_fxn(self,_, x):
        """ODE function to simulate competitive growth.

        Parameters
        ----------
        _ :
            Placeholder for t (not used in this ODE).
        x : type
            Vector of species values at a given time t.

        Returns
        -------
        np.array
            Vector of expected dy_dt values at a given time.

        """

        dx_dt = self.r_i * x *(1-np.sum(x))
        return dx_dt

    def __stop_sim(t,x):

        # Set max val parameter (how close you have to be to a sum of 1 to stop integration)
        max_val = 0.99
        # Calculate sum of all species
        sum = np.sum(x)
        # integrate function terminates when the return is 0
        return (max_val - sum)

    def sim_growth(self, **kwargs):
        """Growth ODE simulation. Terminates when sum of all species is close
        to 1 (So we can set an arbitrarily large t_span).

        Parameters
        ----------
        **kwargs :
            Additional arguments to pass to integration function.
        """

        # Make a stop simulation function when max OD is near 1
        def stop_sim(t,x):
            # Set max val parameter (how close you have to be to a sum of 1 to stop integration)
            max_val = 0.999
            # Calculate sum of all species
            sum = np.sum(x)
            # integrate function terminates when the return is 0
            return (max_val - sum)
        # Set this to be a terminal function
        stop_sim.terminal = True

        # Run the ODE solver
        sol = spi.solve_ivp(self.__ode_fxn, t_span = (0,100000), y0 = self.x_i0_dil,
            events = stop_sim, **kwargs)

        # Set time point array on object
        self.t = sol.t
        # Set x(t) matrix on object
        self.x_t = sol.y

        # Run the enrichment calculation
        self.__calc_enrich()

    def __calc_enrich(self):
        """ Calculate fold enrichment values from original library distribution after
        outgrowth.

        """
        # Get final values of library members
        final_xs = self.x_t[:,-1]
        self.enrichs = final_xs / self.x_i0

    def plot_x_t(self, ax, **kwargs):
        """Plotting function to plot all species in a run.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            Matplotlib axis.

        """
        for x in self.x_t[:,:]:
            ax.plot(self.t, x, **kwargs)
        ax.set_ylim(0,1)
