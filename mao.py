import os
import torch


class MAOdataset(torch.utils.data.Dataset):
    def __init__(self, path_to_folder):
        super(MAOdataset).__init__()
        self.path = path_to_folder
        self.dataset_lenth = 0

        while os.path.isfile(self.path + "molecule" + str(self.dataset_lenth) + ".ct"):
            self.dataset_lenth += 1


    def __getitem__(self, index):
       
        if index < 0 or index >= self.dataset_lenth:
            raise KeyError ("Dataset index out of bound")
        else:

            dsfile = open (self.path + "dataset.ds", "r")

            y = int(dsfile.readlines()[index].split()[1])

            f = open (self.path + "molecule" + str(index) + ".ct", "r")
            f.readline()

            sizes = f.readline().split()

            V_n = int(sizes[0])
            E_n = int(sizes[1]) 

            E = torch.zeros((V_n, 6), dtype=torch.float)

            for i in range(V_n):
                line = f.readline().split()

                attributes = [float(line[i]) for i in range(len(line) - 1)]

                #Replacing the elements with one-hot encoding
                if line[len(line) - 1] == 'N':
                    attributes = attributes + [1., 0., 0.]
                elif line[len(line) - 1] == 'C':
                    attributes = attributes + [0., 1., 0.]
                elif line[len(line) - 1] == 'O':
                    attributes = attributes + [0., 0., 1.]
                else:
                    attributes = attributes + [0., 0., 0.]

                E[i] = torch.tensor(attributes, dtype=torch.float)

            V_E = torch.zeros((E_n, 2), dtype=torch.int)
            V = torch.zeros((E_n, 6), dtype=torch.float)
            for i in range(E_n):
                line = f.readline().split()
                
                line = [int(s) for s in line]

                V_E[i] = torch.tensor(line[0:2], dtype=torch.int)
                
                attributes = []
                if line[2] == 1:
                    attributes = attributes + [1., 0., 0.]
                elif line[2] == 2:
                    attributes = attributes + [0., 1., 0.]
                elif line[2] == 3:
                    attributes = attributes + [0., 0., 1.]
                else:
                    attributes = attributes + [0., 0., 0.]

                if line[3] == 1:
                    attributes = attributes + [1., 0., 0.]
                elif line[3] == 2:
                    attributes = attributes + [0., 1., 0.]
                elif line[3] == 3:
                    attributes = attributes + [0., 0., 1.]
                else:
                    attributes = attributes + [0., 0., 0.]    

                
                V[i] = torch.tensor(attributes, dtype=torch.float)
            return (E, V_E, V), torch.tensor(float(y), dtype=torch.float)

    def __len__(self):
        return self.dataset_lenth
 

