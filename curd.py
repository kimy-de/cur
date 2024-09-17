import numpy as np
from sklearn.utils.extmath import randomized_svd

__all__ = ["CURdecomposition"]

class CURdecomposition:
    def __init__(self, A=None, n_components=2, sampling='deim', n_iter=10):
        """Dimensionality reduction using CUR decomposition

        Parameters
        ----------
        A: numpy.array, default=None, n x m dataset matrix (n: number of features, m: number of instances)
        n_components : int, default=2, Desired dimensionality of output data
        sampling : str, default='deim', Sampling method for column and row selection
        n_iter : int, default=10, Number of iterations for CUR approximations when sampling='random'

        Attributes
        ----------
        C : numpy.array, Matrix containing selected columns of A
        Cpinv : numpy.array, Pseudo inverse of C
        R : numpy.array, Matrix containing selected rows of A    
        Rpinv : numpy.array, Pseudo inverse of R
        U : numpy.array, U = Cpinv @ A @ Rpinv
        colidx : list, List of selected column indices for constructing C
        rowidx : list, List of selected row indices for constructing R

        Notes
        -----
        CUR decomposition is sensitive to column and row selection. 
        Thus, various sampling methods will be explored for optimal selection.

        """ 

        self.A = A 
        self.n_components = n_components
        self.n_iter = n_iter

        if sampling == 'random':
            self.C, self.U, self.R, self.Cpinv, self.Rpinv, self.colidx, self.rowidx = self.random_selection()

        elif sampling == 'deim':
            # Truncated SVD
            U, sigma, VT = randomized_svd(self.A, n_components=n_components, random_state=77)
            
            self.colidx = self.deim(VT.T)
            self.rowidx = self.deim(U)
            self.C = self.A[:,self.colidx]
            self.R = self.A[self.rowidx]
            self.Cpinv = self.inverse(self.C)
            self.Rpinv = self.inverse(self.R)
            self.U = self.Cpinv @ self.A @ self.Rpinv


    def deim(self, V):
        #print(V.shape)
        p = []
        v = V[:,0]
        p.append(np.argmax(abs(v)))

        for i in range(1,V.shape[1]):
            v = V[:,i:i+1]
            c = self.inverse(V[p,:i]) @ v[p]
            r = v - V[:,:i] @ c
            p.append(np.argmax(abs(r)))

        return p

    def random_selection(self):
        """Column and row selection using multinomial distributions based on data magnitudes.

        Returns
        -------
        C : numpy.array, Matrix containing selected columns of A
        Cpinv : numpy.array, Pseudo inverse of C
        R : numpy.array, Matrix containing selected rows of A    
        Rpinv : numpy.array, Pseudo inverse of R
        U : numpy.array, U = Cpinv @ A @ Rpinv
        colidx : list, List of selected column indices for constructing C
        rowidx : list, List of selected row indices for constructing R
        
        """

        e = 1e+10 # dummy error

        for i in range(self.n_iter):
            # column selection
            p1 = np.sum(self.A**2, axis=0)/np.sum(self.A**2) # m probabilities
            colidx = np.random.choice(len(p1), size=self.n_components, p=p1, replace=False) 

            # row selection
            p2 = np.sum(self.A**2, axis=1)/np.sum(self.A**2) # m probabilities
            rowidx = np.random.choice(len(p2), size=self.n_components, p=p2, replace=False) 

            # CUR decomposition
            C = self.A[:,colidx]
            R = self.A[rowidx]
            Cpinv = self.inverse(C)
            Rpinv = self.inverse(R)
            U = Cpinv @ self.A @ Rpinv
            error = np.linalg.norm(self.A - C @ U @ R) # L2 norm

            # best choice of a CUR approximation
            if error < e:
                e = error
                cur_set = (C, U, R, Cpinv, Rpinv, colidx, rowidx)
        
        #print(f"[CUR decomposition] approximation error: {e:.4f}")

        return cur_set[0], cur_set[1], cur_set[2], cur_set[3], cur_set[4], cur_set[5], cur_set[6]

    def inverse(self, X):
        """Inverse of X

        Parameters
        ----------
        X : numpy.array, Matrix

        Returns
        -------
        inv : (pseudo)inverse of X

        """

        try:
            inv = np.linalg.inv(X)
            return inv
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(X)
            return inv

