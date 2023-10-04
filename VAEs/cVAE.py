# BASIC EXPLANATION: https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a

# PAPER: https://arxiv.org/pdf/1711.00937.pdf
# SAMPLE VQ-VAE: https://github.com/airalcorn2/vqvae-pytorch/blob/master/train_vqvae.py

# CODE cond VAE: https://github.com/chendaichao/VAE-pytorch/blob/master/Models/VAE/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# Set the main device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reparameterize_gaussian(mean, std):
    """
    Inputs:
        mean : [torch.tensor] Mean vector. Shape: batch_size x z_dim.
        std  : [torch.tensor] Standard deviation vection. Shape: batch_size x z_dim.
    
    Output:
        z    : [torch.tensor] z sampled from the Normal distribution with mean and standard deviation given by the inputs. 
                              Shape: batch_size x z_dim.
    """

    # Sample epsilon from N(0,I)
    eps = torch.randn_like(std)

    # Calculate z using reparameterization trick
    z = mean + std*eps

    return z


class CEncoder(nn.Module):
    """ Convolutional encoder for the CVAE. """

    def __init__(self, z_dim, n_classes, n_channels):
        super().__init__()

        feature_dim = 32 * 6 * 6  # FIXME PROB HERE IS THE ISSUE
        self.conv1 = nn.Conv2d(n_channels + 1, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.mean_fc = nn.Linear(feature_dim, z_dim)
        self.logvar_fc = nn.Linear(feature_dim, z_dim)
        self.cls_token_fc = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        
        mean = self.mean_fc(x)  # mat1 and mat2 shapes cannot be multiplied (64x22816 and 1152x2)
        logvar = self.logvar_fc(x)
        cls_token = self.cls_token_fc(x)

        return mean, logvar, cls_token
    
    
class CDecoder(nn.Module):
    """ Convolutional decoder for the CVAE. """

    def __init__(self, z_dim, n_classes, n_channels):
        super().__init__()
        
        feature_dim = 32 * 6 * 6  # FIXME PROB HERE IS THE ISSUE
        self.fc = nn.Linear(z_dim + n_classes, feature_dim)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
        self.conv1 = nn.ConvTranspose2d(16, n_channels, kernel_size=3, stride=2, output_padding=1)

    def forward(self, z):
        # print("dec z.size()", z.size())
        x = F.relu(self.fc(z))
        x = x.view(-1, 32, 6, 6)
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        # print("dec x.size()", x.size())
        
        return x
    
    
class CVAE(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, z_dim, n_classes, n_channels=1, img_size=[28,28]):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_size = img_size

        self.encoder = CEncoder(z_dim, n_classes, n_channels)
        self.decoder = CDecoder(z_dim, n_classes, n_channels)

        # Add learnable class token
        self.cls_param = nn.Parameter(torch.zeros(n_classes, *img_size))

    def get_cls_emb(self, c):
        return self.cls_param[c].unsqueeze(1)

    def forward(self, x, c):
        """
        Args:
            x   : [torch.Tensor] Image input of shape [batch_size, n_channels, *img_size]
            c   : [torch.Tensor] Class labels for x of shape [batch_size], where the class in indicated by a
        """

#        assert x.shape[1:] == (self.n_channels, *self.img_size), \
#            f'Expected input x of shape [batch_size, {[self.n_channels, *self.img_size]}], but got {x.shape}'
#        assert c.shape[0] == x.shape[0], \
#            f'Inputs x and c must have same batch size, but got {x.shape[0]} and {c.shape[0]}'
#        assert len(c.shape) == 1, \
#            f'Input c should have shape [batch_size], but got {c.shape}'

        # Get cls embedding
        cls_emb = self.get_cls_emb(c)

        # Concatenate cls embedding to the input
        x = torch.cat((x, cls_emb), dim=1)

        # Get the mean, logvar, and cls token from the encoder
        mean, logvar, cls_token = self.encoder(x)

        # Calculate the standard deviation. Note: in log-space, squareroot is divide by two
        std = torch.exp(logvar / 2)

        # Sample the latent using the reparameterization trick
        z = reparameterize_gaussian(mean, std)
        # print("######")
        # print("z.size() =", z.size())
        
        # Concatenate cls token to z
        z = torch.cat((z, F.softmax(cls_token, dim=1)), dim=1)
        
        # Get reconstructed x from the decoder
        x_hat = self.decoder(z)
        
        return x, x_hat, mean, logvar, cls_token
    