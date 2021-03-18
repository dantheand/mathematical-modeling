""" All the nested objects required to simulate cell circuits in a plate setting.

TODO: Convert all the iterations over the lattice into some sort of object or function.
- iterator object?
- pass it the function to be executed in each well?
- create a quicker way to make plates in a loop (overarching function)
- create overarching function to plot inducers and cell states at the same time
- allow model to take non-binary values (repressor function output)
    - make lin classifier model a class that extends a parental model class
    - add other types of models
        - graded outputs
        - convert one input to be non-linear via sigmoid
"""

import numpy as np
import weakref
from IPython.core.debugger import set_trace
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class Diffusor:

    def __init__(self, name, coordinates = (0,0), diff_c = 1):
        """ Inititalize w/ (0,0) coordinates and diffusion constant = 1

        Parameters
        ----------
        name: string
            UNIQUE name for inducer
        coordinates: tuple, optional
            x,y coordinates of inducer centroid.
        diff_c: float, optional
            Diffusion constant for the inducer
        """

        self.coord = coordinates
        self.diff_c = diff_c
        self.name = name

    def conc_fnc(self,x_val,y_val):
        """ Provide concetration at a given x/y coordinate given object attributes """

        z = 1/(4*np.pi*self.diff_c)*np.exp(
            -((x_val - self.coord[0])**2 + (y_val - self.coord[1])**2)/(4*self.diff_c)
            )
        return z


    def get_conc(self,x_vec,y_vec):
        """ Get concentrations for a given set of x-y points

        Takes a list of x,y coordinate (vectors) and returns the concentration values
        at those points. Vectorizes the conc_fnc function.

        """
        z_vals = []
        vfunc = np.vectorize(self.conc_fnc)

        z_vals = np.array(vfunc(x_vec,y_vec))
        return z_vals

class Plate:

    def __init__(self, dim = (10,5), well_dim = (200,100)):
        """ Inititalize a plate and create wells.

        Parameters
        ----------
        dim: tuple, optional
            x,y overall dimensions of the plate.
        well_dim: tuple, optional
            Number of wells along each x,y axis.

        """
        # Create empty lattice to fill with well objects.
        self.lattice = np.empty(well_dim, dtype=object)

        # Create meshgrid object to get coordinates for each well
        self.x_sites = np.linspace(0,dim[0], well_dim[0]) #access these when plotting
        self.y_sites = np.linspace(0,dim[1], well_dim[1])
        xs, ys = np.meshgrid(self.x_sites,self.y_sites, indexing = 'xy', sparse=True)

        # Combine xs and ys into single array with tuples for coordinates
        coord_arr = np.empty((well_dim[0],well_dim[1]), dtype=tuple)
        for i in range(self.lattice.shape[0]):
            for j in range(self.lattice.shape[1]):
               # Create a coordinate array
                coord_arr[i,j] = (xs[0,i], ys[j,0])
                self.lattice[i,j] = Well(coords = coord_arr[i,j])

    def apply_inducer(self,inducer):
        """ Apply an inducer over all of the wells.

        Parameters
        ----------
        inducer: Diffusor object
            Inducer object to be added to well.
        """
        # Copy and paste code from above to iterate over all lattice locations
        for i in range(self.lattice.shape[0]):
            for j in range(self.lattice.shape[1]):

               # Get Well and query inducer for concentration at that coordinate
               well = self.lattice[i,j]
               ind_conc = inducer.conc_fnc(well.coord[0], well.coord[1])
               # Apply add inducer method from well object
               well.add_inducer(ind_name = inducer.name, ind_val = ind_conc)

    def add_cells(self,model, loc = 'all'):
        """ Instantiates and adds cells with a given model to the plate positions

        Currently can only add cells to all coordinates of the plate.

        Parameters
        ----------
        model: Lin_classifier object
            classifier object within the cells.
        """

        if loc != 'all':
            raise ValueError('Sorry... feature to add cells to other locations not implemented yet')
        else:
            # Copy and paste code from above to iterate over all lattice locations
            for i in range(self.lattice.shape[0]):
                for j in range(self.lattice.shape[1]):
                    # Get well
                    well = self.lattice[i,j]
                    # Instantiate cells for this well and pass it the model
                    cells = Cells(model = model,parent = well)
                    # give it to the well
                    well.add_cells(cells)

    def plot_ind(self, ax, param_dict = {}):
        """ Plot inducer values over the plate.

        Parameters
        ----------
        ax: Axes object
            Matplotlib axes object to plot onto.
        param_dict: Dictionary, optional
            Dictionary with extra parameters to be passed to the plot function.

        TODO: Figure out how to properly set alpha levels for different numbers of
            inducers.
        """

        # Generate inducer values matrix for each well; first axis is inducer ID
        #   First pull inducer names and number from first well.
        ind_dict = self.lattice[0,0].ind_dict
        # Note reverse indexing for array
        z_ind_vals = np.empty((len(ind_dict),len(self.y_sites),len(self.x_sites)))
        # Iterate over all inducers at all lattice locations
        ind_i = 0
        for ind_name in ind_dict.keys():
            for i in range(self.lattice.shape[0]):
                for j in range(self.lattice.shape[1]):

                   # Get Well and query for inducer concentration for given inducer
                   well = self.lattice[i,j]
                   z_ind_vals[ind_i, j, i] = well.ind_dict[ind_name]
            # Iterate inducer index
            ind_i = ind_i + 1

        # Create contourf plot for all inducers
        cmap_list = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges'] #colormaps to pull from

        # iterate through all inducers and plot each
        for i in range(len(ind_dict)):
            ax.contourf(self.x_sites,self.y_sites, z_ind_vals[i],
                cmap = cmap_list[i], alpha = 0.3, **param_dict)

        return ax

    def plot_cells(self, ax, param_dict = {}):
        """ Plot cell states over the plate.

        Parameters
        ----------
        ax: Axes object
            Matplotlib axes object to plot onto.
        param_dict: Dictionary, optional
            Dictionary with extra parameters to be passed to the plot function.
        """
        # Create empty array to store cells (note reversed indexing)
        cell_vals = np.empty((len(self.y_sites),len(self.x_sites)))
        # Iterate over lattice
        for i in range(self.lattice.shape[0]):
            for j in range(self.lattice.shape[1]):
               # Get Well and query cells in well for their state
               cells = self.lattice[i,j].cells

               # Convert boolean to float value
               boolean_val = cells.circuit_val
               if boolean_val:
                   float_val = 1
               else:
                   float_val = 0

               # Add to z_vals
               cell_vals[j, i] = float_val

        # Plot it
        #set_trace()
        ax.contourf(self.x_sites,self.y_sites, cell_vals, **param_dict)
        return ax



class Well:
    """ Well class which contains cells and belongs to Plate. """

    def __init__(self, coords):
        """ Initialize a plate well with its coordinates and inducer values

        Parameters
        ----------
        coords: tuple
            x,y coordinates of well on plate.
        """
        self.coord = coords

        # Instantiate a inducer values dictionary to hold various inducer concentrations
        #   key = inducer name, value = inducer concentration
        self.ind_dict = {}

    def add_inducer(self, ind_name, ind_val):
        """ Add inducer concentration value to well.

        Parameters
        ----------
        ind_name: string
            Name of the inducer.
        ind_val: float
            concentration value of inducer at the well.
        """

        # Change inducer value for given inducer name
        self.ind_dict[ind_name] = ind_val

    def add_cells(self,cells):
        """ Add cells attribute to well. """
        self.cells = cells

    def report_inducer(self):
        """ Report back inducer values as key-valued pair """
        return self.ind_dict

class Cells:
    """ Class that contains the 'circuit' (model) that computes inducer / diffusor levels """

    def __init__(self, parent, model):
        self.model = model
        self.coord= parent.coord # pass it the parent well coordinates
        self.ind = parent.ind_dict # get inducer values from parent

        # Also set state of cell (ON or OFF) depending on well's inducer concentrations
        ind_vals = list(self.ind.values())
        # Trim inducer values to length of model if they're longer
        ind_vals = ind_vals[0:model.n_dim]

        self.circuit_val = self.model.return_state(ind_vals)
        """TODO: figure out how to store the well data with the cells ^^^"""



class Lin_classifier:
    """ Linear classifier model class. """

    def __init__(self, weights, bias):

        """ Inititalization functions

        Parameters
        ----------
        weights: numpy.array
            Weight vector (defines dimensions of classifier)
        bias: float
            Bias value for classifier

        """
        self.weights = np.array(weights)
        self.n_dim = self.weights.shape[0]
        self.bias = bias

    def z_func_3d(self,x_val,y_val):
        """ 3D Z solution to be vectorized. """
        z = -self.weights[0]/self.weights[2]*x_val - \
            self.weights[1]/self.weights[2]*y_val \
            + self.bias/self.weights[2]

        return z


    def return_state(self,coord):
        """ Returns classifier results (binary)

        Parameters
        ----------
        coord: tuple
            x,y coordinates to calculate linear classification for. These should be inducer concentrtions
        """

        if np.dot(self.weights,coord) - self.bias > 0:
            return True
        else:
            return False

    def plot(self, ax, x_range, y_range = [], param_dict={}):
        """ Plot function for the model to axis. Can only do 2D and 3D plots.

        Parameters
        ----------
        ax: Axes object
        x_range: array
            1D Range of x values to plot
        y_range: array, optional
            1D Range of y values to plot. Required for 3D plot
        param_dict: Dictionary, optional
        """
        #2D plot
        if self.n_dim == 2:
            ys = []
            for x in x_range:
                y = - self.weights[0]/self.weights[1]*x + (self.bias/self.weights[1])
                ys.append(y)
            out = ax.plot(x_range,ys, **param_dict)

        elif self.n_dim == 3:
            z_vect_fnc = np.vectorize(self.z_func_3d)
            # Create a meshgrid to pass to vectorized functions
            xs, ys = np.meshgrid(x_range,y_range, sparse = True, indexing = 'xy')
            zs = z_vect_fnc(xs,ys)
            out = ax.plot_surface(xs,ys,zs,
                cmap='viridis', edgecolor='none', rcount = 5, ccount = 5)
        else:
            raise ValueError('Can only plot linear classifiers for 2D and 3D models!')

        return out
