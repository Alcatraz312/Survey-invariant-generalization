import torch
import torch.nn as nn
import torch.nn.functional as F

class MTLArchitecture(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        '''
        input_dim  : number of wavelength pixels -> int (3800)
        latent_dim : dimension of latent space   -> int
        num_classes: number of MK classes        -> int
        '''
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
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

        # Learned global log variance for reconstruction
        # scalar — shared across all pixels
        # avoids conflating encoder uncertainty with reconstruction uncertainty
    

        # Regression head (heteroscedastic)
        # predicts mean + uncertainty for each atmospheric parameter
        self.regression_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.reg_mu     = nn.Linear(32, 3)   # Teff, log g, [Fe/H]
        self.reg_logvar = nn.Linear(32, 3)   # per-parameter log variance

        # Classification head 
        self.classification_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
        # learnable homoscedastic uncertainties for each loss
        self.log_sigma_recon = nn.Parameter(torch.zeros(1)) 
        self.logvar_reg = nn.Parameter(torch.zeros(1))
        self.logvar_cls = nn.Parameter(torch.zeros(1))

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

        # Regression head
        z_reg       = self.regression_head(mu)
        mu_reg      = self.reg_mu(z_reg)       # (batch, 3)
        logvar_reg  = self.reg_logvar(z_reg)   # (batch, 3)

        # Classification head
        cls_logits = self.classification_head(mu)   # (batch, num_classes)

        return {
            "x_hat"      : x_hat,        # reconstructed spectrum
            "mu"         : mu,           # latent mean
            "logvar"     : logvar,       # latent log variance
            "mu_reg"     : mu_reg,       # predicted atmospheric params
            "logvar_reg" : logvar_reg,   # predicted param uncertainties
            "cls_logits" : cls_logits    # raw class logits
        }

    # Loss components

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
        return nll.mean()                            # mean over batch → scalar

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
    
    def uncertainty_aggregate_loss(self,x, x_hat, mu, logvar, mu_reg, logvar_reg, y, cls_logits, labels, beta = 1.0):
        '''
        Aggregated loss \n
        negative Evidence lower bound + regression loss + classification loss
        '''

        recon_loss = self.reconstruction_loss(x, x_hat)
        kl = self.kl_divergence(mu, logvar)
        reg_loss = self.regression_loss(mu_reg, logvar_reg, y)
        cls_loss = self.classification_loss(cls_logits, labels)

        # uncertainty weights for each loss function
        w_recon = 0.5 * torch.exp(-self.log_sigma_recon)
        w_cls = 0.5 * torch.exp(-self.logvar_cls)
        w_reg = 0.5 * torch.exp(-self.logvar_reg)

        # uncertainty weights loss aggregation
        loss = (w_recon * recon_loss + 0.5 * self.log_sigma_recon + beta * kl) + (w_reg * reg_loss + 0.5 * self.logvar_reg) + (w_cls * cls_loss + 0.5 * self.logvar_cls)

        # loss = recon_loss + beta * kl + reg_loss + cls_loss

        #geometric loss aggregation
        # log_geo = (recon_loss + 5.0 * reg_loss + 5.0 * cls_loss)
        # geo_loss = torch.exp(log_geo)
        # loss = geo_loss + beta * kl

        components = {
            "loss" : loss.item(),
            "recon" : recon_loss.item(),
            "kl" : kl.item(),
            "reg" : reg_loss.item(),
            "cls" : cls_loss.item(),

            "sigma_recon" : torch.exp(0.5 * self.log_sigma_recon).item(),
            "sigma_reg"  : torch.exp(0.5 * self.logvar_reg).item(),
            "sigma_cls"  : torch.exp(0.5 * self.logvar_cls).item()
        }

        return loss, components
    
    def sum_aggregate_loss(self, x, x_hat, mu, logvar, mu_reg, logvar_reg, y, cls_logits, labels, beta = 1.0):

        recon_loss = self.reconstruction_loss(x, x_hat)
        kl = self.kl_divergence(mu, logvar)
        reg_loss = self.regression_loss(mu_reg, logvar_reg, y)
        cls_loss = self.classification_loss(cls_logits, labels)

        loss = (recon_loss) + beta * kl + (reg_loss) + (cls_loss)

        # loss with reconstruction loss

        # loss = beta * kl + reg_loss + cls_loss

        components = {
            "loss" : loss.item(),
            "recon" : recon_loss.item(),
            "kl" : kl.item(),
            "reg" : reg_loss.item(),
            "cls" : cls_loss.item(),

            # "sigma_recon" : torch.exp(0.5 * self.logvar_recon).item(),
            # "sigma_reg"  : torch.exp(0.5 * self.logvar_reg).item(),
            # "sigma_cls"  : torch.exp(0.5 * self.logvar_cls).item()
        }

        return loss, components

    def geometric_loss_aggregation(self, x, x_hat, mu, logvar, mu_reg, logvar_reg, y, cls_logits, labels, beta = 1.0):

        recon_loss = self.reconstruction_loss(x, x_hat)
        kl = self.kl_divergence(mu, logvar)
        reg_loss = self.regression_loss(mu_reg, logvar_reg, y)
        cls_loss = self.classification_loss(cls_logits, labels)

        geometric_loss = torch.exp(1/3 * ((torch.log(recon_loss)) + (torch.log(reg_loss)) + (torch.log(cls_loss))))

        regularized_geometric_loss = geometric_loss + beta * kl

        components = {
            "loss" : regularized_geometric_loss.item(),
            "recon" : recon_loss.item(),
            "kl" : kl.item(),
            "reg" : reg_loss.item(),
            "cls" : cls_loss.item(),
        }

        return regularized_geometric_loss, components