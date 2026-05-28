import torch
import torch.nn as nn
import torch.nn.functional as F

class MTLArchitecture(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        '''
        input_dim  : number of wavelength pixels -> int (3800)
        latent_dim : dimension of latent space   -> int
        num_classes: number of MK classes        -> int
        linear_reg : if True, regression head is a single linear layer (no activations)
        '''
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),    # batchnormalization 
            nn.ReLU(),
            nn.Linear(2048, 1024),
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

        # Decoder with no batch normalization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
        )

        # Regression Head 
        self.regression_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),   # batchnormalization 
            nn.ReLU(),
            nn.Dropout(0.1),      # dropouts
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.reg_mu     = nn.Linear(32, 3)
        self.reg_logvar = nn.Linear(32, 3)

        # Classification head — unchanged
        self.classification_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # learnable homoscedastic uncertainties
        self.log_sigma_recon = nn.Parameter(torch.zeros(1))
        self.logvar_reg      = nn.Parameter(torch.zeros(1))
        self.logvar_cls      = nn.Parameter(torch.zeros(1))

    def reparameterize(self, mu, logvar):
        '''
        Reparameterize the latent variables for backpropagation \n
        Parameters : \n
        mu : latent mean vector -> Tensor \n
        logvar : latent variance vector -> Tensor \n
        Returns: \n
        z : reparameterized latent vector
    
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        '''
        Forward pass for the archicture \n
        Parameters: \n
        x : data tensor
        '''
        h      = self.encoder(x)
        mu     = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z      = self.reparameterize(mu, logvar)
        x_hat  = self.decoder(z)

        # regression head — linear or nonlinear depending on flag
        z_reg      = self.regression_head(mu)
        mu_reg     = self.reg_mu(z_reg)
        logvar_reg = self.reg_logvar(z_reg)

        cls_logits = self.classification_head(mu)

        return {
            "x_hat"      : x_hat,
            "mu"         : mu,
            "logvar"     : logvar,
            "mu_reg"     : mu_reg,
            "logvar_reg" : logvar_reg,
            "cls_logits" : cls_logits
        }

    def reconstruction_loss(self, x, x_hat):
        log_sigma = torch.clamp(self.log_sigma_recon, -10, 5)
        var       = torch.exp(2 * log_sigma)
        nll = 0.5 * torch.mean(
            2 * log_sigma + (x - x_hat)**2 / var, dim=1
        )
        return nll.mean()

    def kl_divergence(self, mu, logvar):
        logvar = torch.clamp(logvar, -10, 5)
        kl = 0.5 * torch.sum(
            mu**2 + torch.exp(logvar) - 1 - logvar, dim=1
        )
        return kl.mean()

    def regression_loss(self, mu_reg, logvar_reg, y):
        logvar_reg = torch.clamp(logvar_reg, -10, 5)
        nll = 0.5 * torch.mean(
            logvar_reg + (y - mu_reg)**2 / torch.exp(logvar_reg), dim=1
        )
        return nll.mean()

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
    

# smart initialization of weights

def smart_init(model):    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(       # using kaiming he weight initialization technique 
                module.weight,
                nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            # standard batch normalization starting values