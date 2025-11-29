import torch
import gpytorch

class MixedGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_latents=3, num_tasks=2):
        """
        Multi-output GP using Linear Model of Coregionalization (LMC).
        
        Args:
            inducing_points: Inducing point locations (M x D)
            num_latents: Number of latent functions (default: 3)
            num_tasks: Number of output tasks (default: 2)
        """
        # Variational distribution for latent functions
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), 
            batch_shape=torch.Size([num_latents])
        )
        
        # Base variational strategy
        base_variational_strategy = gpytorch.variational.VariationalStrategy(
            self, 
            inducing_points, 
            variational_distribution, 
            learn_inducing_locations=True
        )
        
        # LMC variational strategy - models task correlations
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            base_variational_strategy,
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )
        
        super().__init__(variational_strategy)
        
        # Mean and covariance for latent functions
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
