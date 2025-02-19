import torch
from random import shuffle

class PointCloudDisplay():

    def __init__(self, M1 , M2) -> None:
        self.M1 = M1
        self.M2 = M2


    def display_val(self, nb_cells, GT, pred, cp, K = 10): # K is the number of points displayed along the tangents
        
        tot_cells, nb_points, _ = GT.shape
        idx = [i for i in range(tot_cells)]
        shuffle(idx)
        idx = idx[:nb_cells]

        clouds_list = list()
        M1, M2 = self.M1, self.M2

        for i in idx:
            crt_GT = torch.concatenate((GT[i], torch.ones((nb_points, 1))),dim=-1)
            crt_pred = torch.concatenate((pred[i], 2*torch.ones((nb_points, 1))),dim=-1)
            reg_cp = torch.concatenate((cp[i,:M1*(M2-1)], 4*torch.ones((M1*(M2-1), 1))),dim=-1)

            poles = cp[i,M1*(M2-1):M1*(M2-1)+2]
            
            T_N = (poles[0][None,None,:] + cp[i,M1*(M2-1)+2:M1*(M2-1)+4].unsqueeze(1)*torch.linspace(0,1,K)[None,:,None]).reshape(-1,3)
            T_S = (poles[1][None,None,:] + cp[i,M1*(M2-1)+4:M1*(M2-1)+6].unsqueeze(1)*torch.linspace(0,1,K)[None,:,None]).reshape(-1,3)

            grad = torch.concatenate((torch.concatenate((T_N,T_S), dim=0),9*torch.ones(4*K,1)),dim=-1)

            clouds_list.append(torch.concatenate((crt_GT,crt_pred, reg_cp, grad), dim=0).numpy())

        return clouds_list
            
