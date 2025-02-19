import torch
from math import cos, sin, pi

def spherical_to_xyz(free_parameters):

    nb_free_params = free_parameters.shape[0]//3
    
    r = free_parameters[:nb_free_params]
    theta = free_parameters[nb_free_params:2*nb_free_params]*torch.pi
    delta = free_parameters[2*nb_free_params:3*nb_free_params]*torch.pi/2

    x = r*torch.cos(delta)*torch.cos(theta)
    y = r*torch.cos(delta)*torch.sin(theta)
    z = r*torch.sin(delta)

    return x, y, z


def a(M):
    return 2*(1-cos(2*pi/M))/(cos(pi/M)-cos(3*pi/M))

def init_sphere(M1, M2):
    a1 = a(M1)
    a2 = a(2*M2)

    k = torch.arange(start=0, end=M1)
    l = torch.arange(start=1, end=M2)

    x = a1*torch.cos(2*torch.pi*k/M1)[:,None]*a2*torch.sin(torch.pi*l/M2)[None,:]
    y = a1*torch.sin(2*torch.pi*k/M1)[:,None]*a2*torch.sin(torch.pi*l/M2)[None,:]
    z = a2*torch.cos(torch.pi*l/M2).repeat(M1,1)

    reg_cp = torch.stack((x,y,z)).permute(1,2,0).reshape(-1,3)

    phi_0 = (cos(pi/(2*M2))- cos(pi/M2))/(1-cos(pi/M2))
    phi_1 = sin(pi/(4*M2))**2/(1-cos(pi/M2))

    cp_N = torch.tensor([[0,0,a2*(phi_0+2*phi_1*cos(pi/M2))]])
    cp_S = torch.tensor([[0,0,-a2*(phi_0+2*phi_1*cos(pi/M2))]])

    phi_prime_1 = 2*cos(pi/(4*M2))*sin(pi/(4*M2))/(1-cos(pi/M2))
    d = pi*a2*phi_prime_1*sin(pi/M2)

    T_N1 = torch.tensor([[d,0,0]])
    T_N2 = torch.tensor([[0,d,0]])
    T_S1 = torch.tensor([[-d,0,0]])
    T_S2 = torch.tensor([[0,-d,0]])

    return torch.concatenate((reg_cp,cp_N,cp_S,T_N1,T_N2,T_S1,T_S2))


class ControlPointsDimmer():
    def __init__(self, M1, M2, device = "cpu") -> None:
        self.M1 = M1 
        self.M2 = M2

        self.nb_free_params = M1*(M2-1)+6

        self.device = device

        self.sphere_cp = init_sphere(M1,M2).unsqueeze(0).to(device)


    def get_sphere(self):
        return self.sphere_cp
    
    def dim_control_points(self, features):
        dx = features[:self.nb_free_params]
        dy = features[self.nb_free_params:2*self.nb_free_params]
        dz = features[2*self.nb_free_params:3*self.nb_free_params]
        r = features[-1]

        d = torch.stack((dx,dy,dz)).permute(2,1,0)

        return self.sphere_cp*r[:,None,None] + d