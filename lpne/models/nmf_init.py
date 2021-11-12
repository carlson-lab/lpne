"""
Initalize a model with NMF.

"""
__date__ = "November 2021"


from sklearn.decomposition import NMF
import torch
import torch.nn.functional as F

NMF_KWARGS = dict(
    init='nndsvdar',
    solver='mu',
    beta_loss=2.0,
    random_state=42,
)


class QuickFactorModel(torch.nn.Module):

    def __init__(self):
        super(QuickFactorModel, self).__init__()

    def forward(self, X):
        """
        X : [n,f,r,r]
        """
        f1 = F.softplus(self.f_factor)
        f2 = F.softplus(self.r1_factor)
        f3 = F.softplus(self.r2_factor)
        res = X - f1*f2*f3
        return torch.pow(res,2).sum(dim=(1,2,3)).mean()


    def fit(self, X, n_iter=20000):
        """
        X : [n,f,r,r]
        """
        n, f, r = X.shape[:-1]
        self.f_factor = torch.nn.Parameter(torch.randn(n,f,1,1))
        self.r1_factor = torch.nn.Parameter(torch.randn(n,1,r,1))
        self.r2_factor = torch.nn.Parameter(torch.randn(n,1,1,r))

        X = torch.tensor(X, dtype=torch.float)

        optimizer = torch.optim.Adam(self.parameters())
        for iter in range(n_iter):
            self.zero_grad()
            loss = self(X)
            loss.backward()
            optimizer.step()
            if iter % 1000 == 0:
                print(f"iter {iter:04d}, loss: {loss.item():3f}")

        with torch.no_grad():
            self.f_factor_ = F.softplus(self.f_factor.view(1,n*f)).detach().cpu().numpy()
            self.r1_factor_ = F.softplus(self.r1_factor.view(1,n*r)).detach().cpu().numpy()
            self.r2_factor_ = F.softplus(self.r2_factor.view(1,n*r)).detach().cpu().numpy()
        return self



def get_initial_factors(features, z_dim, n_rois, n_freqs):
    """

    """
    print("NMF initialization...")
    flat_features = features.reshape(features.shape[0],-1)
    model = NMF(n_components=z_dim, **NMF_KWARGS).fit(flat_features)
    print("done fitting")
    H = model.components_.reshape(z_dim, n_freqs, n_rois, n_rois)

    model = QuickFactorModel().fit(H)
    return model.f_factor_, model.r1_factor_, model.r2_factor_



if __name__ == '__main__':
    pass


###
