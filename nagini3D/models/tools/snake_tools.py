import torch
from math import pi, cos

def create_interval_mask(t, lower_bound, upper_bound):
    """
        Return a binary with 1 where lower_bound <= t < upper_bound and 0 elsewhere.
    """
    return (lower_bound <= t)*(t<upper_bound)*1.


def create_exponential_spline(M):

    def basis_function_t(t):
        abs_t = torch.abs(t)

        mask_0 = create_interval_mask(abs_t, 0, 0.5)
        values_0 = mask_0*(torch.cos(2*pi*abs_t/M)*cos(pi/M) - cos(2*pi/M))

        mask_1 = create_interval_mask(abs_t, 0.5, 1.5)
        values_1 = mask_1*(torch.sin(pi*(3/2-abs_t)/M)*torch.sin(pi*(3/2-abs_t)/M))
        
        phi_M_t = 1/(1-cos(2*pi/M))*(values_0 + values_1)

        return phi_M_t

    return basis_function_t


def create_periodic_exponential_spline(M):

    basis_fct_aux = create_exponential_spline(M)

    # sufficient to use this support because for other indices the function is zero everywhere
    n_min = -1
    n_max = 1


    def basis_periodic_function(t):
        phi_t = 0
        for n in range(n_min,n_max+1):
            phi_t += basis_fct_aux(t-M*n)
        return phi_t

    return basis_periodic_function



def create_polynomials():

    def basis_function(t):
        t_2 = t*t
        mask_0 = create_interval_mask(t, -1, 0)
        values_0 = mask_0*(2-t_2*(3*t+5))

        mask_1 = create_interval_mask(t, 1, 2)
        values_1 = mask_1*(-(t-2)*(t-2)*(t-1))

        mask_2 = create_interval_mask(t, -2, -1)
        values_2 = mask_2*((t+1)*(t+2)*(t+2))

        mask_3 = create_interval_mask(t, 0, 1)
        values_3 = mask_3*((3*t-5)*t_2 + 2)

        phi_t = 0.5*(values_0 + values_1 + values_2 + values_3)

        return phi_t
    
    return basis_function

