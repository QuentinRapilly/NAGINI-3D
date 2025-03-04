import numpy as np
import torch
from math import pi, cos, sin

from .snake_tools import (create_exponential_spline, create_periodic_exponential_spline)

class SnakeSmoothSampler():
    def __init__(self, P, M1, M2=None, device = "cpu"):
        # Creation of Fibonacci Lattice 
        self.P = P
        assert self.P%2 == 1

        self.device = device

        golden_ratio = (np.sqrt(5)-1)/2

        N = P // 2
        i = np.arange(-N, N+1)
        lat = np.arcsin(2*i/P)/torch.pi + 0.5
        long = np.mod(i*golden_ratio, 1)

        self.u = torch.unsqueeze(torch.tensor(long),1) 
        self.v = torch.unsqueeze(torch.tensor(lat), 1)

        # Computation of sampling weights
        self.M1 = M1
        if M2 == None:
            self.M2=M1
        else:
            self.M2=M2

        i = torch.arange(0, self.M1).unsqueeze(0)
        j = torch.arange(-1, self.M2+2).unsqueeze(0)

        # Careful phi_1 = M1*phi but phi_2 = 2*M2*phi

        self.phi_per = create_periodic_exponential_spline(M1)
        self.phi_per_weights = self.phi_per(M1*self.u-i).unsqueeze(2)

        self.phi = create_exponential_spline(2*M2)
        self.phi_weights = self.phi(M2*self.v-j).unsqueeze(1)

        self.weights = (self.phi_per_weights*self.phi_weights).unsqueeze(0)[...,None].to(device)

        gamma = 2*(1-cos(2*pi/M1))/(cos(pi/M1)-cos(3*pi/M1))
        self.cM1 = gamma*torch.cos(2*torch.pi*i/M1)[...,None]
        self.sM1 = gamma*torch.sin(2*torch.pi*i/M1)[...,None]

        self.phi_0 = (cos(pi/(2*M2))-cos(pi/M2))/(1-cos(pi/M2))  #compute_phi_0(M2)
        self.phi_1 = (1-cos((pi)/(2*M2)))/(2*(1-cos(pi/M2))) # compute_phi_1(M2)
        self.phi_prime_1 = (-pi/(2*M2))*sin(pi/(2*M2))/(1-cos(pi/M2)) # compute_phi_prime_1(M2)

        self.fi_weights = self.init_fi_weights(self.phi_per_weights, self.phi_weights)[None,...,None].to(device=self.device)

        #print(f"Valeurs des vecteurs phi, 0 : {self.phi_0}, 1 : {self.phi_1}, 1' : {self.phi_prime_1}")

    def init_fi_weights(self, phi_1, phi_2):
        gamma = self.phi_1/self.phi_0
        Gamma = phi_2[...,2:-2]
        Gamma[...,0] += phi_2[...,0] - 2*gamma*phi_2[...,1]
        Gamma[...,-1] += phi_2[...,-1] - 2*gamma*phi_2[...,-2]

        fi_weights = phi_1*Gamma
        a,b,c = fi_weights.shape
        fi_weights = fi_weights.reshape(a,b*c)
        
        Gamma_N = phi_2[...,1]/self.phi_0
        Gamma_S = phi_2[...,-2]/self.phi_0

        Gamma_1N = phi_2[...,0] - gamma*phi_2[...,1]
        Gamma_1S = gamma*phi_2[...,-2] - phi_2[...,-1]

        alpha_k = self.cM1/(self.M2*self.phi_prime_1)
        beta_k = self.sM1/(self.M2*self.phi_prime_1)

        A_u = (alpha_k*phi_1).sum(dim=(-2))
        B_u = (beta_k*phi_1).sum(dim=(-2))
        
        return torch.cat((fi_weights, Gamma_N, Gamma_S, Gamma_1N*A_u,
                                     Gamma_1N*B_u, Gamma_1S*A_u, Gamma_1S*B_u), dim=-1)
    

    def up_sampling_uv(self, M1, M2):
        i = torch.arange(0, M1*(M2-1))
        k = i//(M2-1)
        l = i%(M2-1) + 1
        u = k/M1
        v = l/M2
        u = torch.cat((u, torch.tensor([0, 0.33, 0.66, 0, 0.33, 0.66])))
        eps = 1/(2*M2)
        v = torch.cat((v, torch.tensor([eps, eps, eps, 1-eps, 1-eps, 1-eps])))
    
    def up_sampling_uv(self, M1, M2):
        i = torch.arange(0, M1*(M2-1))
        k = i//(M2-1)
        l = i%(M2-1) + 1
        u = k/M1
        v = l/M2
        u = torch.cat((u, torch.tensor([0, 0.33, 0.66, 0, 0.33, 0.66])))
        eps = 1/(2*M2)
        v = torch.cat((v, torch.tensor([eps, eps, eps, 1-eps, 1-eps, 1-eps])))

        return u[...,None], v[...,None]

    def up_sample_transfer_matrix(self):
        u, v = self.up_sampling_uv(self.M1, self.M2)

        i = torch.arange(0, self.M1).unsqueeze(0)
        j = torch.arange(-1, self.M2+2).unsqueeze(0)

        phi_1 = self.phi_per(self.M1*u-i).unsqueeze(2)
        phi_2 = self.phi(self.M2*v-j).unsqueeze(1)

        return torch.linalg.inv(self.init_fi_weights(phi_1, phi_2)).to(self.device)
    
    def up_sample_sampling(self, fi, new_M1, new_M2):
        u, v = self.up_sampling_uv(new_M1, new_M2)

        i = torch.arange(0, self.M1).unsqueeze(0)
        j = torch.arange(-1, self.M2+2).unsqueeze(0)

        phi_1 = self.phi_per(self.M1*u-i).unsqueeze(2)
        phi_2 = self.phi(self.M2*v-j).unsqueeze(1)

        weights = self.init_fi_weights(phi_1, phi_2)[None,...,None].to(self.device)

        return (fi[:,None,...]*weights).sum(dim=-2)

    def free_parameters_to_cp(self, free_parameters):

        """
        - control_points: (batch, M1.(M2-1)+6, 3) #dans l'article ils disent +4 mais c'est très bizarre
            The control_points vector is composed of:
                - (M1x(M2-1),3) regular control points
                - (1, 3)  north pole control point
                - (1, 3)  south pole control point
                - (2, 3)  north pole derivative vector
                - (2, 3)  south pole derivative vector
        """
        cM1 = self.cM1.to(self.device)
        sM1 = self.sM1.to(self.device)

        M1, M2 = self.M1, self.M2
        batch_size = free_parameters.shape[0]

        reg_cp = free_parameters[:,:M1*(M2-1)].reshape(batch_size, M1, M2-1, 3)
        cp_N = free_parameters[:,M1*(M2-1)].unsqueeze(-2)
        cp_S = free_parameters[:,M1*(M2-1)+1].unsqueeze(-2)
        T_N1 = free_parameters[:,M1*(M2-1)+2].unsqueeze(-2)
        T_N2 = free_parameters[:,M1*(M2-1)+3].unsqueeze(-2)
        T_S1 = free_parameters[:,M1*(M2-1)+4].unsqueeze(-2)
        T_S2 = free_parameters[:,M1*(M2-1)+5].unsqueeze(-2)

        cp_i_1 = reg_cp[:,:,0]
        cp_i_M2_1 = reg_cp[:,:,-1]

        a = cp_i_1 + (T_N1*cM1 + T_N2*sM1)/(M2*self.phi_prime_1) 
        b = cp_N/self.phi_0  - self.phi_1*(a + cp_i_1)/self.phi_0 # careful, it is a "-"" not a "+" as said in the paper, there is a mistake in the proof

        d = cp_i_M2_1 - (T_S1*cM1 + T_S2*sM1)/(M2*self.phi_prime_1)
        c = cp_S/self.phi_0 - self.phi_1*(cp_i_M2_1 + d)/self.phi_0

        cp = torch.cat((a.unsqueeze(-2),
                        b.unsqueeze(-2),
                        reg_cp,
                        c.unsqueeze(-2),
                        d.unsqueeze(-2)), dim=-2)
        
        return cp.unsqueeze(1)
    

    def get_dckl_dfi(self):
        M1, M2 = self.M1, self.M2
        I = M1*(M2-1) + 6
        mid = torch.zeros((M1*(M2-1)+6, M1, M2-1))
        i = torch.arange(0,I-6)
        k = i//(M2-1)
        l = i%(M2-1)
        mid[i,k,l] = 1

        left = torch.zeros((M1*(M2-1)+6, M1, 2))
        right = torch.zeros((M1*(M2-1)+6, M1, 2))

        alpha = self.phi_1/self.phi_0
        R = (I - 6) - 1 # index where the fi that are not ckl starts 

        cM1 = self.cM1.squeeze()
        sM1 = self.sM1.squeeze()

        k = torch.arange(0, M1)
        left[(M2-1)*k,k,0] = 1
        left[R+3,:,0] = cM1/(M2*self.phi_prime_1)
        left[R+4,:,0] = sM1/(M2*self.phi_prime_1)

        left[(M2-1)*k,k,1] = -2*alpha
        left[R+1,:,0] = 1/self.phi_0
        left[R+3,:,1] = -alpha*cM1/(M2*self.phi_prime_1)
        left[R+4,:,1] = -alpha*sM1/(M2*self.phi_prime_1)

        right[(M2-1)*(k+1)-1,k,1] = 1
        right[R+5,:,1] = -cM1/(M2*self.phi_prime_1)
        right[R+6,:,1] = -sM1/(M2*self.phi_prime_1)

        right[(M2-1)*(k+1)-1,k,1] = -2*alpha
        right[R+2,:,0] = 1/self.phi_0
        right[R+5,:,1] = alpha*cM1/(M2*self.phi_prime_1)
        right[R+6,:,1] = alpha*sM1/(M2*self.phi_prime_1)

        return torch.cat((left,mid,right), dim = -1).to(self.device)

    def sample_snakes(self, fi):
        return (fi[:,None,...]*self.fi_weights).sum(dim=-2)
    

    def create_facets(self, n_u, n_v):
        i, j = torch.arange(n_u)[...,None], torch.arange(1,n_v-1)[None, ...]

        def f(u, v, width : int):
            return u*width+v
        
        s1 = f(i,j,n_v).reshape(n_u*(n_v-2))
        s2 = f(torch.remainder(i+1, n_u), j, n_v).reshape(n_u*(n_v-2))
        s3 = f(i, torch.remainder(j+1, n_v), n_v).reshape(n_u*(n_v-2))
        s4 = f(torch.remainder(i-1, n_u), j, n_v).reshape(n_u*(n_v-2))
        s5 = f(i, torch.remainder(j-1, n_v), n_v).reshape(n_u*(n_v-2))

        facets1 = torch.stack((s1,s2,s3)).T
        facets2 = torch.stack((s1,s4,s5)).T

        return torch.cat((facets1, facets2))
    

    def draw_surface(self, free_parameters : torch.Tensor, points_per_dim : tuple[int, int]):

        cp = self.free_parameters_to_cp(free_parameters=free_parameters)
        n_u, n_v = points_per_dim

        u, v = (torch.arange(n_u)/n_u)[...,None], (torch.arange(n_v)/(n_v-1))[...,None]

        i, j = torch.arange(0, self.M1)[None,...], torch.arange(-1, self.M2+2)[None, ...]

        phi_u = self.phi_per(self.M1*u-i)[:,None,:,None]
        phi_v = self.phi(self.M2*v-j)[None,:,None,:]

        weights = (phi_u*phi_v).reshape(n_u*n_v, self.M1, self.M2+3)[None,...,None].to(self.device)

        points = (weights*cp).sum(dim=(-2,-3))

        facets = self.create_facets(n_u,n_v)

        values = (torch.cos(2*torch.pi*u)+torch.cos(torch.pi*v).squeeze()[None,...]).reshape(-1)

        return points, facets, values

    
if __name__ == "__main__":

    M = 3
    M1 = 2*M
    M2 = M
    sampler = SnakeSmoothSampler(P = 51, M1 = M1, M2 = M2)

    #cp = torch.rand((1,M1*(M2-1)+6, 3))
    #sampling = sampler.sample_snakes(cp)

    sphere = sampler.sample_sphere()
    #print(sampling.shape)
