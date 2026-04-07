import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Latent distribution
        self.mu_layer     = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

        self.log_sigma_recon = nn.Parameter(torch.zeros(1))
        

    # Reparameterization
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # forward pass
    def forward(self, x):
        # Encode
        h      = self.encoder(x)
        mu     = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        # Sample latent vector
        z = self.reparameterize(mu, logvar)

        # Decode
        x_hat = self.decoder(z)

        return {
            "x_hat"      : x_hat,        # reconstructed spectrum
            "mu"         : mu,           # latent mean
            "logvar"     : logvar,       # latent log variance
        }

    def reconstruction_loss(self, x, x_hat):
        '''
        Gaussian NLL with learned global variance
        log_sigma_recon is a learned scalar parameter
        mean over pixels, mean over batch → scalar
        '''
        log_sigma = torch.clamp(self.log_sigma_recon, -10, 5)
        var       = torch.exp(2 * log_sigma)

        nll = 0.5 * torch.mean(
            2 * log_sigma + (x - x_hat)**2 / var,
            dim=1                                    # mean over pixels
        )
        return nll.mean() 
    
    def kl_divergence(self, mu, logvar):
        '''
        KL( q(z|x) || N(0,I) )
        sum over latent dims, mean over batch → scalar
        '''
        logvar = torch.clamp(logvar, -10, 5)
        kl = 0.5 * torch.sum(
            mu**2 + torch.exp(logvar) - 1 - logvar,
            dim=1                                    # sum over latent dims
        )
        return kl.mean()                             # mean over batch → scalar
    

class DownHeads(VAE):

    def __init__(self, input_dim, latent_dim, num_classes = 8):
        super().__init__(input_dim, latent_dim)

        # Regression head (heteroscedastic)
        # predicts mean + uncertainty for each atmospheric parameter
        self.regression_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.reg_mu     = nn.Linear(32, 3)   # Teff, log g, [Fe/H]
        self.reg_logvar = nn.Linear(32, 3)   # per-parameter log variance

        self.classification_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.logvar_cls = nn.Parameter(torch.zeros(1))
        self.logvar_reg = nn.Parameter(torch.zeros(1))

    
    def forward(self, x):
    # Get VAE outputs from parent
        out = super().forward(x)

        # Regression head takes latent mean as input
        z_reg      = self.regression_head(out["mu"])
        mu_reg     = self.reg_mu(z_reg)       # (batch, 3)
        logvar_reg = self.reg_logvar(z_reg)   # (batch, 3)

        cls_logits = self.classification_head(out["mu"])

        # Return everything
        return {
            "x_hat"      : out["x_hat"],
            "mu"         : out["mu"],
            "logvar"     : out["logvar"],
            "mu_reg"     : mu_reg,
            "logvar_reg" : logvar_reg,
            "cls_logits" : cls_logits
        }
    
    def regression_loss(self, mu_reg, logvar_reg, y):
        '''
        Heteroscedastic Gaussian NLL for atmospheric parameters
        y     : ground truth params (batch, 3)
        mean over params, mean over batch → scalar
        '''
        logvar_reg = torch.clamp(logvar_reg, -10, 5)
        nll = 0.5 * torch.mean(
            logvar_reg + (y - mu_reg)**2 / torch.exp(logvar_reg),
            dim=1                                    # mean over 3 params
        )
        return nll.mean()                            # mean over batch → scalar
    
    def classification_loss(self, cls_logits, labels):
        '''
        Standard cross entropy
        labels : integer class indices (batch,)
        '''
        return F.cross_entropy(cls_logits, labels)
    
