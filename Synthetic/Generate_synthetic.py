import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import torch.distributions as dis

def calculate_SCIR(S, C):
    shared_energy = np.linalg.norm(S, ord="fro") / S.shape[0]
    private_energy = np.mean([np.linalg.norm(p, ord="fro") / p.shape[0] for p in C])
    scir = 10 * np.log10(shared_energy / private_energy)

    return scir

def generate_mixture_of_gaussians(dim, num_samples, mu, sigma, num_components, isotropic= True):
    # print(num_components)
    # Initialize the parameters of the mixture model
    weights = torch.ones(num_components) / num_components
    means = torch.randn(num_components, dim)*5
    
    if isotropic == True:
        covariances = torch.diag(torch.abs(torch.randn(dim)*2) ).repeat(num_components, 1, 1)
    else:
        covariances = torch.vstack([torch.diag(torch.abs(torch.randn(dim)*2) ) for _ in range(num_components)]).reshape(num_components, dim, dim)

    # print(means, covariances)

    # Create a categorical distribution to sample the components
    categorical = dis.Categorical(weights)

    # Generate samples from the mixture of Gaussians
    samples = torch.zeros(num_samples, dim)
    component_indices = categorical.sample((num_samples,))
    for i in range(num_components):
        mask = component_indices == i
        num_masked = mask.sum()
        if num_masked > 0:
            component_samples = dis.MultivariateNormal(means[i], covariances[i]).sample((num_masked,))
            samples[mask] = component_samples
            # print(samples[mask].shape)
            
    samples =  sigma * (samples - torch.mean(samples)) / torch.std(samples) + mu
    return samples.T

class ViewDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        self.data_len = data_tensor.shape[1]

    def __getitem__(self, index):
        return self.data[:, index]

    def __len__(self):
        return self.data_len


class GenerateData:
    def __init__(self, dim_info: dict, batch_size) -> None:
        self.N = dim_info["N"]
        self.D = dim_info["D"]
        self.D1 = dim_info["D1"]
        self.D2 = dim_info["D2"]
        self.M1 = dim_info["M1"]
        self.M2 = dim_info["M2"]

        self.batch_size = batch_size

    def generate_matrix_with_condition_number(self, m, n, mu, sigma, dist, condition_number):

        A = self.generate_distribution(m, n, mu, sigma, dist)
        U, S, V = torch.svd(A)
        S = S - torch.min(S)
        S = (S * ((condition_number-1)/ torch.max(S) ) ) + 1
        A = torch.matmul(U, torch.matmul(torch.diag(S), V.t()))

        return A

    def generate_distribution(self, m, n, mu, sigma, dist):

        if dist == "normal":
            randvar = sigma * torch.randn(m, n) + mu
            return randvar
        
        elif dist.startswith("mixture"):
            params = dist.split("_")
            num_comp = int(params[-1])
            if params[1] == "iso":
                isotropic = True
            else:
                isotropic = False

            randvar = generate_mixture_of_gaussians(m, n, mu, sigma, num_comp, isotropic=isotropic)
            return randvar
            
        elif dist == "uniform":
            scale_factor = np.sqrt(3)*sigma
            randvar = -2*scale_factor*torch.rand(m, n) + mu + scale_factor
            
            return randvar
            
        elif dist == "laplace":
            d = dis.laplace.Laplace(torch.tensor([float(mu)]), torch.tensor([sigma/np.sqrt(2)]))
            val = d.sample(sample_shape=(m, n)).squeeze(dim=-1)
            randvar = val.type(torch.FloatTensor)
            return randvar
            
        elif dist == "gamma":
            d = dis.gamma.Gamma(torch.tensor([float(mu**2/ sigma**2)]), torch.tensor([mu/sigma**2]))
            val = d.sample(sample_shape=(m, n)).squeeze(dim=-1)
            randvar = val.type(torch.FloatTensor)
            return randvar
            
        elif dist == "beta":
            d = dis.beta.Beta(torch.tensor([float(mu)]), torch.tensor([float(sigma)]))
            val = d.sample(sample_shape=(m, n)).squeeze(dim=-1)
            randvar = val.type(torch.FloatTensor)
            return randvar
            
        elif dist == "vonmises":
            d = dis.von_mises.VonMises(torch.tensor([float(sigma)]), torch.tensor([float(mu)]))
            val = d.sample(sample_shape=(m, n)).squeeze(dim=-1)
            randvar = val.type(torch.FloatTensor)
            return randvar
        else:
            raise Exception("Distribution not recognized.")

    def generate_cca_data(self, signal, noise, mixer, normalize_mean=False, diff_mixture=True):
        assert self.N >= self.D + self.D1 + self.D2, "T matrix not full column rank."
        assert self.M1 >= self.D + self.D1, "Mixing matrix 1 is not full column rank."
        assert self.M2 >= self.D + self.D2, "Mixing matrix 2 is not full column rank."
        assert self.N >= self.M1, "Data is not full column rank."

        dist_S = signal["dist"]
        dist_C1 = noise["dist1"]
        dist_C2 = noise["dist2"]

        if isinstance(dist_S, list):
            if len(dist_S) > 1: 
                S = torch.cat([self.generate_distribution(1, self.N, signal["mean"], signal["std"], dist_each) for dist_each in dist_S], dim=0)
            else:
                S = self.generate_distribution(self.D, self.N, signal["mean"], signal["std"], dist_S[0])
        else:
            S = self.generate_distribution(self.D, self.N, signal["mean"], signal["std"], dist_S)
            

        # Mixing matrices
        print("Generating mixing  matrix.")
        A_1 = self.generate_matrix_with_condition_number(self.M1, self.D + self.D1, mixer["mean"], mixer["std"], mixer["dist"], condition_number = 5.0)
        while torch.linalg.cond(A_1) > 5.0:
            A_1 = self.generate_matrix_with_condition_number(self.M1, self.D + self.D1, mixer["mean"], mixer["std"], mixer["dist"], condition_number = 5.0)

        self.A1 = A_1.clone()

        if diff_mixture:
            A_2 = self.generate_matrix_with_condition_number(self.M2, self.D + self.D2, mixer["mean"], mixer["std"], mixer["dist"], condition_number = 5.0)
            while torch.linalg.cond(A_2) > 5.0:
                A_2 = self.generate_matrix_with_condition_number(self.M2, self.D + self.D2, mixer["mean"], mixer["std"], mixer["dist"], condition_number = 5.0)
        else:
            A_2 = A_1.clone()

        self.A2 = A_2.clone()

        if self.D1 and self.D2:
            # With private Component

            # Private component of view 1 and view 2
            C_1 = self.generate_distribution(self.D1, self.N, noise["mean1"], noise["std1"], dist_C1)
            C_2 = self.generate_distribution(self.D2, self.N, noise["mean2"], noise["std2"], dist_C2)

            # Generate data using LMM
            Z1 = torch.vstack((S, C_1))
            Z2 = torch.vstack((S, C_2))
            X_1 = torch.matmul(A_1, Z1)
            X_2 = torch.matmul(A_2, Z2)

            # Calculate Shared Component to Interference Ratio
            scir = calculate_SCIR(S.numpy(), [C_1.numpy(), C_2.numpy()])

        else:
            #No private Component

            # Generate data using LMM
            X_1 = torch.matmul(A_1, S)
            X_2 = torch.matmul(A_2, S)

            scir = signal["std"]**2

        
        # Normalize the column of data to make zero mean
        if normalize_mean:
            X_1 = X_1 - torch.mean(X_1, 1, keepdim=True)
            X_2 = X_2 - torch.mean(X_2, 1, keepdim=True)


        assert np.linalg.matrix_rank(X_1) == self.D + self.D1, "First View matrix is not full rank."
        assert np.linalg.matrix_rank(X_2) == self.D + self.D2, "Second View matrix is not full rank."

        print(f"Shared Component to Interference Ratio: {scir: .3f}")
        
        return X_1, X_2, S, scir, C_1, C_2

    def get_mixing_matrices(self):
        return self.A1, self.A2

    def create_dataloader(self, X_1, X_2):

        v1 = ViewDataset(X_1)
        v2 = ViewDataset(X_2)

        view1 = torch.utils.data.DataLoader(
            v1, batch_size=self.batch_size, shuffle=True
        )

        view2 = torch.utils.data.DataLoader(
            v2, batch_size=self.batch_size, shuffle=True
        )

        return view1, view2