from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as networkx
import numpy as np
import scipy as scipy
import scipy.integrate

# This work is a modification of code from https://github.com/ryansmcgee/seirsplus.
class NetworkModel():
    """
    A class to simulate the Stochastic Network Model
    ===================================================
    Params: 
            # Networks on which transmission occurs:
            G  Network adjacency matrix (np array) or Networkx graph object.
            Q  Quarantine adjacency matrix (np array) or Networkx graph object.

            # Rate of transmission:
            beta_I  Rate of transmission from symptomatic individuals
            beta_A  Rate of transmission from asymptomatic individuals
            beta_DI Rate of transmission for individuals with detected symptomatic infections
            beta_DA Rate of transmission for individuals with detected asymptomatic infections

            # Probabilities of being asymptomatic and locality parameter
            p_a  Probability of being an asymptomatic carrier
            p    Probability of interaction outside adjacent nodes

            # Rate of Infections:
            sigma   Rate of infection (upon exposure) 
            sigma_D Rate of infection (upon exposure) for individuals with detected infections

            # Rate of recovery
            gamma_I  Rate of recovery (upon infection) 
            gamma_A  Rate of recovery (upon infection) 
            gamma_DI Rate of recovery (upon infection) for individuals with detected symptomatic infections
            gamma_DA Rate of recovery (upon infection) for individuals with detected asymptomatic infections

            # Rates of infection related deaths:
            mu_I  Rate of infection-related death
            mu_A  Rate of infection-related death
            mu_DI 
            mu_DA 

            xi Rate of re-susceptibility (upon recovery)  

            # Parameters that are used for testing:
            phi_E   Rate of contact tracing testing for exposed individuals
            phi_I   Rate of contact tracing testing for symptomatic individuals
            phi_A   Rate of contact tracing testing for asymptomatic individuals
            psi_E   Probability of positive test results for exposed individuals
            psi_I   Probability of positive test results for symptomatic individuals
            psi_A   Probability of positive test results for asymptomatic individuals
            q       Probability of quarantined individuals interaction outside adjacent nodes
            theta_E Rate of baseline testing for exposed individuals
            theta_I Rate of baseline testing for symptomatic individuals
            theta_A Rate of baseline testing for asymptomatic individuals
            
            initE   Init number of exposed individuals       
            initI   Init number of symptomatic individuals      
            initA   Init number of asymptomatic individuals      
            initD_E Init number of detected infectious individuals
            initD_I Init number of detected symptomatic individuals   
            initD_A Init number of detected asymptomatic individuals   
            initR   Init number of recovered individuals     
            initF   Init number of infection-related fatalities
                    (all remaining nodes initialized susceptible)   
    """

    def __init__(self, G, Q = None, beta_I = 0, beta_A = 0, beta_DI = 0, beta_DA = 0,
                 p_a = 0, p = 0,
                 sigma = 0, sigma_D = 0,
                 gamma_I = 0, gamma_A = 0, gamma_DI = 0, gamma_DA = 0,
                 mu_I = 0, mu_A = 0, mu_DI = 0, mu_DA = 0,
                 xi = 0,
                 phi_E = 0, phi_I = 0, phi_A = 0, psi_E = 0, psi_I = 0, psi_A = 0, 
                 q = 0, theta_E = 0, theta_I = 0, theta_A = 0,
                 initE = 0, initI = 0, initA = 0, initD_E = 0, initD_I = 0, initD_A = 0, initR = 0, initF = 0             
                ):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Adjacency matrix:
        self.update_G(G)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup Quarantine Adjacency matrix:
        if(Q is None):
            Q = G # If no Q graph is provided, use G in its place
        self.update_Q(Q)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.parameters = {'beta_I':beta_I, 'beta_A':beta_A, 'beta_DI':beta_DI, 'beta_DA':beta_DA,
                           'p_a':p_a, 'p':p, 
                           'sigma':sigma, 'sigma_D':sigma_D,
                           'gamma_I':gamma_I, 'gamma_A':gamma_A, 'gamma_DI':gamma_DI, 'gamma_DA':gamma_DA,
                           'mu_I':mu_I, 'mu_A':mu_A, 'mu_DI':mu_DI, 'mu_DA':mu_DA,
                           'xi':xi, 
                           'phi_E':phi_E, 'phi_I':phi_I, 'phi_A':phi_A, 'psi_E':psi_E, 'psi_I':psi_I, 'psi_A':psi_A,
                           'q':q, 'theta_E':theta_E, 'theta_I':theta_I, 'theta_A':theta_A,
                           'initE':initE, 'initI':initI, 'initA':initA, 'initD_E':initD_E, 'initD_I':initD_I, 
                           'initD_A':initD_A, 'initR':initR, 'initF':initF
                          }
        self.update_parameters()

        # Initialization:
        self.tseries = np.zeros(5*self.numNodes)
        self.numE    = np.zeros(5*self.numNodes)
        self.numI    = np.zeros(5*self.numNodes)
        self.numA    = np.zeros(5*self.numNodes)
        self.numD_E  = np.zeros(5*self.numNodes)
        self.numD_I  = np.zeros(5*self.numNodes)
        self.numD_A  = np.zeros(5*self.numNodes)
        self.numR    = np.zeros(5*self.numNodes)
        self.numF    = np.zeros(5*self.numNodes)
        self.numS    = np.zeros(5*self.numNodes)
        self.N       = np.zeros(5*self.numNodes)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t          = 0
        self.tmax       = 0 # will be set when run() is called
        self.tidx       = 0
        self.tseries[0] = 0
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.numE[0]   = int(initE)
        self.numI[0]   = int(initI)
        self.numA[0]   = int(initI)
        self.numD_E[0] = int(initD_E)
        self.numD_I[0] = int(initD_I)
        self.numD_A[0] = int(initD_I)
        self.numR[0]   = int(initR)
        self.numF[0]   = int(initF)
        self.numS[0]   =   self.numNodes - self.numE[0] - self.numI[0] - self.numA[0] - self.numD_E[0] - self.numD_I[0] - self.numD_A[0] \
                         - self.numR[0] - self.numF[0]
        self.N[0]      = self.numS[0] + self.numE[0] + self.numI[0] + self.numA[0] + self.numD_E[0] + self.numD_I[0] + self.numD_A[0] + self.numR[0]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Node states:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.S   = 1
        self.E   = 2
        self.I   = 3
        self.A   = 4
        self.D_E = 5
        self.D_I = 6
        self.D_A = 7
        self.R   = 8
        self.F   = 9

        # Initializing the state vector X:
        self.X = np.array([self.S]*int(self.numS[0]) + 
                          [self.E]*int(self.numE[0]) + 
                          [self.I]*int(self.numI[0]) + 
                          [self.A]*int(self.numA[0]) + 
                          [self.D_E]*int(self.numD_E[0]) + 
                          [self.D_I]*int(self.numD_I[0]) + 
                          [self.D_A]*int(self.numD_A[0]) + 
                          [self.R]*int(self.numR[0]) + 
                          [self.F]*int(self.numF[0])).reshape((self.numNodes,1))
        np.random.shuffle(self.X)

        # UNUSED FEATURE:
        self.store_Xseries = False
        
        self.transitions =  { 
                                'StoE': {'currentState':self.S, 'newState':self.E},
                                'EtoI': {'currentState':self.E, 'newState':self.I},
                                'EtoA': {'currentState':self.E, 'newState':self.A},
                                'ItoR': {'currentState':self.I, 'newState':self.R},
                                'AtoR': {'currentState':self.A, 'newState':self.R},
                                'ItoF': {'currentState':self.I, 'newState':self.F},
                                'AtoF': {'currentState':self.A, 'newState':self.F},
                                'RtoS': {'currentState':self.R, 'newState':self.S},
                                'EtoDE': {'currentState':self.E, 'newState':self.D_E},
                                'ItoDI': {'currentState':self.I, 'newState':self.D_I},
                                'AtoDA': {'currentState':self.A, 'newState':self.D_A},
                                'DEtoDI': {'currentState':self.D_E, 'newState':self.D_I},
                                'DEtoDA': {'currentState':self.D_E, 'newState':self.D_A},
                                'DItoR': {'currentState':self.D_I, 'newState':self.R},
                                'DAtoR': {'currentState':self.D_A, 'newState':self.R},
                                'DItoF': {'currentState':self.D_I, 'newState':self.F},
                                'DAtoF': {'currentState':self.D_A, 'newState':self.F},
                            }


    def update_parameters(self):
        import time
        updatestart = time.time()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta_I  = np.full(fill_value=self.parameters['beta_I'], shape=(self.numNodes,1))
        self.beta_A  = np.full(fill_value=self.parameters['beta_A'], shape=(self.numNodes,1))
        self.beta_DI  = np.full(fill_value=self.parameters['beta_DI'], shape=(self.numNodes,1))
        self.beta_DA  = np.full(fill_value=self.parameters['beta_DA'], shape=(self.numNodes,1))

        self.p_a = np.full(fill_value=self.parameters['p_a'], shape=(self.numNodes,1))
        self.p   = np.full(fill_value=self.parameters['p'], shape=(self.numNodes,1))

        self.sigma   = np.full(fill_value=self.parameters['sigma'], shape=(self.numNodes,1))
        self.sigma_D = np.full(fill_value=self.parameters['sigma_D'], shape=(self.numNodes,1))

        self.gamma_I  = np.full(fill_value=self.parameters['gamma_I'], shape=(self.numNodes,1))
        self.gamma_A  = np.full(fill_value=self.parameters['gamma_A'], shape=(self.numNodes,1))
        self.gamma_DI = np.full(fill_value=self.parameters['gamma_DI'], shape=(self.numNodes,1))
        self.gamma_DA = np.full(fill_value=self.parameters['gamma_DA'], shape=(self.numNodes,1))

        self.mu_I = np.full(fill_value=self.parameters['mu_I'], shape=(self.numNodes,1))
        self.mu_A = np.full(fill_value=self.parameters['mu_A'], shape=(self.numNodes,1))
        self.mu_DI = np.full(fill_value=self.parameters['mu_DI'], shape=(self.numNodes,1))
        self.mu_DA = np.full(fill_value=self.parameters['mu_DA'], shape=(self.numNodes,1))

        self.xi = np.full(fill_value=self.parameters['xi'], shape=(self.numNodes,1))

        self.phi_E = np.full(fill_value=self.parameters['phi_E'], shape=(self.numNodes,1))
        self.phi_I = np.full(fill_value=self.parameters['phi_I'], shape=(self.numNodes,1))
        self.phi_A = np.full(fill_value=self.parameters['phi_A'], shape=(self.numNodes,1))
        self.psi_E = np.full(fill_value=self.parameters['psi_E'], shape=(self.numNodes,1))
        self.psi_I = np.full(fill_value=self.parameters['psi_I'], shape=(self.numNodes,1))
        self.psi_A = np.full(fill_value=self.parameters['psi_A'], shape=(self.numNodes,1))

        self.q       = np.full(fill_value=self.parameters['q'], shape=(self.numNodes,1))
        self.theta_E = np.full(fill_value=self.parameters['theta_E'], shape=(self.numNodes,1))
        self.theta_I = np.full(fill_value=self.parameters['theta_I'], shape=(self.numNodes,1))
        self.theta_A = np.full(fill_value=self.parameters['theta_A'], shape=(self.numNodes,1))

        self.beta_I_local  = self.beta_I
        self.beta_A_local  = self.beta_A
        self.beta_DI_local = self.beta_DI
        self.beta_DA_local = self.beta_DA
        
        # Pre-multiply beta values by the adjacency matrix ("transmission weight connections")
        if(self.beta_I_local.ndim == 1):
            self.A_beta_I = scipy.sparse.csr_matrix.multiply(self.A_G, np.tile(self.beta_I_local, (1,self.numNodes))).tocsr()
        elif(self.beta_I_local.ndim == 2):
            self.A_beta_I = scipy.sparse.csr_matrix.multiply(self.A_G, self.beta_I_local).tocsr()

        if(self.beta_A_local.ndim == 1):
            self.A_beta_A = scipy.sparse.csr_matrix.multiply(self.A_G, np.tile(self.beta_A_local, (1,self.numNodes))).tocsr()
        elif(self.beta_A_local.ndim == 2):
            self.A_beta_A = scipy.sparse.csr_matrix.multiply(self.A_G, self.beta_A_local).tocsr()

        # Pre-multiply beta_D values by the quarantine adjacency matrix ("transmission weight connections")
        if(self.beta_DI_local.ndim == 1):
            self.A_Q_beta_DI = scipy.sparse.csr_matrix.multiply(self.A_Q, np.tile(self.beta_DI_local, (1,self.numNodes))).tocsr()
        elif(self.beta_DI_local.ndim == 2):
            self.A_Q_beta_DI = scipy.sparse.csr_matrix.multiply(self.A_Q, self.beta_DI_local).tocsr()

        if(self.beta_DA_local.ndim == 1):
            self.A_Q_beta_DA = scipy.sparse.csr_matrix.multiply(self.A_Q, np.tile(self.beta_DA_local, (1,self.numNodes))).tocsr()
        elif(self.beta_DA_local.ndim == 2):
            self.A_Q_beta_DA = scipy.sparse.csr_matrix.multiply(self.A_Q, self.beta_DA_local).tocsr()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update scenario flags:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.update_scenario_flags()

    def node_degrees(self, Amat):
        return Amat.sum(axis=0).reshape(self.numNodes,1)   # sums of adj matrix cols

    def update_G(self, new_G):
        self.G = new_G
        # Adjacency matrix:
        if type(new_G)==np.ndarray:
            self.A_G = scipy.sparse.csr_matrix(new_G)
        elif type(new_G)==networkx.classes.graph.Graph:
            self.A_G = networkx.adj_matrix(new_G) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes = int(self.A_G.shape[1])
        self.degree   = np.asarray(self.node_degrees(self.A_G)).astype(float)

        return

    def update_Q(self, new_Q):
        self.Q = new_Q
        # Quarantine Adjacency matrix:
        if type(new_Q)==np.ndarray:
            self.A_Q = scipy.sparse.csr_matrix(new_Q)
        elif type(new_Q)==networkx.classes.graph.Graph:
            self.A_Q = networkx.adj_matrix(new_Q) # adj_matrix gives scipy.sparse csr_matrix
        else:
            raise BaseException("Input an adjacency matrix or networkx object only.")

        self.numNodes_Q   = int(self.A_Q.shape[1])
        self.degree_Q     = np.asarray(self.node_degrees(self.A_Q)).astype(float)

        assert(self.numNodes == self.numNodes_Q), "The normal and quarantine adjacency graphs must be of the same size."

        return

    def update_scenario_flags(self):
        self.testing_scenario   = ( (np.any(self.psi_I) and (np.any(self.theta_I) or np.any(self.phi_I)))  
                                    or (np.any(self.psi_E) and (np.any(self.theta_E) or np.any(self.phi_E))) 
                                    or (np.any(self.psi_A) and (np.any(self.theta_A) or np.any(self.phi_A)))   
                                  )
        self.tracing_scenario   = ( (np.any(self.psi_E) and np.any(self.phi_E)) 
                                    or (np.any(self.psi_I) and np.any(self.phi_I)) 
                                    or (np.any(self.psi_A) and np.any(self.phi_A)) 
                                  )
        self.resusceptibility_scenario  = (np.any(self.xi))

    def total_num_infections(self, t_idx=None):
        if(t_idx is None):
            return (self.numE[:] + self.numI[:] + self.numA[:] + self.numD_E[:] + self.numD_I[:] + self.numD_A[:])            
        else:
            return (self.numE[t_idx] + self.numI[t_idx] + self.numA[t_idx] + self.numD_E[t_idx] + self.numD_I[t_idx] + self.numD_A[t_idx])          

    def calc_propensities(self):
        # Pre-calculate matrix multiplication terms that may be used in multiple propensity calculations,
        # and check to see if their computation is necessary before doing the multiplication
        transmissionTerms_I = np.zeros(shape=(self.numNodes,1))
        if(np.any(self.numI[self.tidx]) 
            and np.any(self.beta_I!=0)):
            transmissionTerms_I = np.asarray( scipy.sparse.csr_matrix.dot(self.A_beta_I, self.X==self.I) )

        transmissionTerms_A = np.zeros(shape=(self.numNodes,1))
        if(np.any(self.numA[self.tidx]) 
            and np.any(self.beta_A!=0)):
            transmissionTerms_A = np.asarray( scipy.sparse.csr_matrix.dot(self.A_beta_A, self.X==self.A) )

        transmissionTerms_DI = np.zeros(shape=(self.numNodes,1))
        if(self.testing_scenario
            and np.any(self.numD_I[self.tidx])
            and np.any(self.beta_DI)):
            transmissionTerms_DI = np.asarray( scipy.sparse.csr_matrix.dot(self.A_Q_beta_DI, self.X==self.D_I) )

        transmissionTerms_DA = np.zeros(shape=(self.numNodes,1))
        if(self.testing_scenario
            and np.any(self.numD_A[self.tidx])
            and np.any(self.beta_DA)):
            transmissionTerms_DA = np.asarray( scipy.sparse.csr_matrix.dot(self.A_Q_beta_DA, self.X==self.D_A) )

        numContacts_D = np.zeros(shape=(self.numNodes,1))
        if(self.tracing_scenario 
            and (np.any(self.numD_E[self.tidx]) or np.any(self.numD_I[self.tidx]) or np.any(self.numD_A[self.tidx]))):
            numContacts_D = np.asarray( scipy.sparse.csr_matrix.dot(self.A_G, ((self.X==self.D_E)|(self.X==self.D_I)|(self.X==self.D_A)) ) )
                                            
        propensities_StoE   = ( self.p*((self.beta_I*self.numI[self.tidx] + self.beta_A*self.numA[self.tidx] + self.q*self.beta_DI*self.numD_I[self.tidx] + self.q*self.beta_DA*self.numD_A[self.tidx])/self.N[self.tidx])
                                + (1-self.p)*np.divide((transmissionTerms_I + transmissionTerms_A + transmissionTerms_DI + transmissionTerms_DA), self.degree, out=np.zeros_like(self.degree), where=self.degree!=0)
                              )*(self.X==self.S)

        propensities_EtoI   = (1 - self.p_a) * self.sigma*(self.X==self.E)
        propensities_EtoA   = self.p_a * self.sigma*(self.X==self.E)
        propensities_ItoR   = self.gamma_I*(self.X==self.I)
        propensities_AtoR   = self.gamma_A*(self.X==self.A)
        propensities_ItoF   = self.mu_I*(self.X==self.I)
        propensities_AtoF   = self.mu_A*(self.X==self.A)
        propensities_EtoDE  = (self.theta_E + self.phi_E*numContacts_D)*self.psi_E*(self.X==self.E)
        propensities_ItoDI  = (self.theta_I + self.phi_I*numContacts_D)*self.psi_I*(self.X==self.I)
        propensities_AtoDA  = (self.theta_A + self.phi_A*numContacts_D)*self.psi_A*(self.X==self.A)
        propensities_DEtoDI = (1 - self.p_a) * self.sigma_D*(self.X==self.D_E)
        propensities_DEtoDA = self.p_a * self.sigma_D*(self.X==self.D_E)
        propensities_DItoR  = self.gamma_DI * (self.X==self.D_I)
        propensities_DItoF  = self.mu_DI * (self.X==self.D_I)
        propensities_DAtoR  = self.gamma_DA * (self.X==self.D_A)
        propensities_DAtoF  = self.mu_DA * (self.X==self.D_A)
        propensities_RtoS   = self.xi*(self.X==self.R)

        propensities = np.hstack([propensities_StoE, propensities_EtoI, propensities_EtoA,
                                  propensities_ItoR, propensities_AtoR, propensities_ItoF, propensities_AtoF,
                                  propensities_EtoDE, propensities_ItoDI, propensities_AtoDA, propensities_DEtoDI, propensities_DEtoDA,
                                  propensities_DItoR, propensities_DItoF, propensities_DAtoR, propensities_DAtoF,
                                  propensities_RtoS])

        columns = ['StoE', 'EtoI', 'EtoA', 'ItoR', 'AtoR', 'ItoF', 'AtoF', 'EtoDE', 'ItoDI', 'AtoDA', 'DEtoDI', 'DEtoDA', 
                   'DItoR', 'DItoF', 'DAtoR', 'DAtoF', 'RtoS']

        return propensities, columns

    def increase_data_series_length(self):
        self.tseries= np.pad(self.tseries, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numS   = np.pad(self.numS, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numE   = np.pad(self.numE, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numI   = np.pad(self.numI, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numA   = np.pad(self.numA, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_E = np.pad(self.numD_E, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_I = np.pad(self.numD_I, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numD_A = np.pad(self.numD_A, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numR   = np.pad(self.numR, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.numF   = np.pad(self.numF, [(0, 5*self.numNodes)], mode='constant', constant_values=0)
        self.N      = np.pad(self.N, [(0, 5*self.numNodes)], mode='constant', constant_values=0)

        if(self.store_Xseries):
            self.Xseries = np.pad(self.Xseries, [(0, 5*self.numNodes), (0,0)], mode=constant, constant_values=0)

        return None

    def finalize_data_series(self):
        self.tseries= np.array(self.tseries, dtype=float)[:self.tidx+1]
        self.numS   = np.array(self.numS, dtype=float)[:self.tidx+1]
        self.numE   = np.array(self.numE, dtype=float)[:self.tidx+1]
        self.numI   = np.array(self.numI, dtype=float)[:self.tidx+1]
        self.numA   = np.array(self.numA, dtype=float)[:self.tidx+1]
        self.numD_E = np.array(self.numD_E, dtype=float)[:self.tidx+1]
        self.numD_I = np.array(self.numD_I, dtype=float)[:self.tidx+1]
        self.numD_A = np.array(self.numD_A, dtype=float)[:self.tidx+1]
        self.numR   = np.array(self.numR, dtype=float)[:self.tidx+1]
        self.numF   = np.array(self.numF, dtype=float)[:self.tidx+1]
        self.N      = np.array(self.N, dtype=float)[:self.tidx+1]

        if(self.store_Xseries):
            self.Xseries = self.Xseries[:self.tidx+1, :]

        return None

    def run_iteration(self):

        if(self.tidx >= len(self.tseries)-1):
            # Room has run out in the timeseries storage arrays; double the size of these arrays:
            self.increase_data_series_length()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Generate 2 random numbers uniformly distributed in (0,1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        r1 = np.random.rand()
        r2 = np.random.rand()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2. Calculate propensities
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities, transitionTypes = self.calc_propensities()

        # Terminate when probability of all events is 0:
        if(propensities.sum() <= 0.0):            
            self.finalize_data_series()
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 3. Calculate alpha
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        propensities_flat   = propensities.ravel(order='F')
        cumsum              = propensities_flat.cumsum()
        alpha               = propensities_flat.sum()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 4. Compute the time until the next event takes place
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tau = (1/alpha)*np.log(float(1/r1))
        self.t += tau

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 5. Compute which event takes place
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        transitionIdx   = np.searchsorted(cumsum,r2*alpha)
        transitionNode  = transitionIdx % self.numNodes
        transitionType  = transitionTypes[ int(transitionIdx/self.numNodes) ]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 6. Update node states and data series
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert(self.X[transitionNode] == self.transitions[transitionType]['currentState'] and self.X[transitionNode]!=self.F), "Assertion error: Node "+str(transitionNode)+" has unexpected current state "+str(self.X[transitionNode])+" given the intended transition of "+str(transitionType)+"."
        self.X[transitionNode] = self.transitions[transitionType]['newState']

        self.tidx += 1
        
        self.tseries[self.tidx]  = self.t
        self.numS[self.tidx]     = np.clip(np.count_nonzero(self.X==self.S), a_min=0, a_max=self.numNodes)
        self.numE[self.tidx]     = np.clip(np.count_nonzero(self.X==self.E), a_min=0, a_max=self.numNodes)
        self.numI[self.tidx]     = np.clip(np.count_nonzero(self.X==self.I), a_min=0, a_max=self.numNodes)
        self.numA[self.tidx]     = np.clip(np.count_nonzero(self.X==self.A), a_min=0, a_max=self.numNodes)
        self.numD_E[self.tidx]   = np.clip(np.count_nonzero(self.X==self.D_E), a_min=0, a_max=self.numNodes)
        self.numD_I[self.tidx]   = np.clip(np.count_nonzero(self.X==self.D_I), a_min=0, a_max=self.numNodes)
        self.numD_A[self.tidx]   = np.clip(np.count_nonzero(self.X==self.D_A), a_min=0, a_max=self.numNodes)
        self.numR[self.tidx]     = np.clip(np.count_nonzero(self.X==self.R), a_min=0, a_max=self.numNodes)
        self.numF[self.tidx]     = np.clip(np.count_nonzero(self.X==self.F), a_min=0, a_max=self.numNodes)
        self.N[self.tidx]        = np.clip((self.numS[self.tidx] + self.numE[self.tidx] + self.numI[self.tidx]  + self.numA[self.tidx] + self.numD_E[self.tidx] + self.numD_I[self.tidx] + self.numD_A[self.tidx] + self.numR[self.tidx]), a_min=0, a_max=self.numNodes)

        if(self.store_Xseries):
            self.Xseries[self.tidx,:] = self.X.T

        # Termination condition:
        if(self.t >= self.tmax or (self.numI[self.tidx]<1 and self.numE[self.tidx]<1 and self.numD_E[self.tidx]<1 and self.numD_I[self.tidx]<1)):
            self.finalize_data_series()
            return False

        return True


    def run(self, T, checkpoints=None, print_interval=10, verbose='t'):
        if(T>0):
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(checkpoints):
            numCheckpoints = len(checkpoints['t'])
            for chkpt_param, chkpt_values in checkpoints.items():
                assert(isinstance(chkpt_values, (list, np.ndarray)) and len(chkpt_values)==numCheckpoints), "Expecting a list of values with length equal to number of checkpoint times ("+str(numCheckpoints)+") for each checkpoint parameter."
            checkpointIdx  = np.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
            if(checkpointIdx >= numCheckpoints):
                # We are out of checkpoints, stop checking them:
                checkpoints = None 
            else:
                checkpointTime = checkpoints['t'][checkpointIdx]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print_reset = True
        running     = True
        while running:

            running = self.run_iteration()

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Handle checkpoints if applicable:
            if(checkpoints):
                if(self.t >= checkpointTime):
                    if(verbose is not False):
                        print("[Checkpoint: Updating parameters]")
                    # A checkpoint has been reached, update param values:
                    if('G' in list(checkpoints.keys())):
                        self.update_G(checkpoints['G'][checkpointIdx])
                    if('Q' in list(checkpoints.keys())):
                        self.update_Q(checkpoints['Q'][checkpointIdx])
                    for param in list(self.parameters.keys()):
                        if(param in list(checkpoints.keys())):
                            self.parameters.update({param: checkpoints[param][checkpointIdx]})
                    # Update parameter data structures and scenario flags:
                    self.update_parameters()
                    # Update the next checkpoint time:
                    checkpointIdx  = np.searchsorted(checkpoints['t'], self.t) # Finds 1st index in list greater than given val
                    if(checkpointIdx >= numCheckpoints):
                        # We are out of checkpoints, stop checking them:
                        checkpoints = None 
                    else:
                        checkpointTime = checkpoints['t'][checkpointIdx]

            if(print_interval):
                if(print_reset and (int(self.t) % print_interval == 0)):
                    if(verbose=="t"):
                        print("t = %.2f" % self.t)
                    print_reset = False
                elif(not print_reset and (int(self.t) % 10 != 0)):
                    print_reset = True

        return True

    def plot(self, ax=None,  plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                            plot_D_E='line', plot_D_I='line', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True):

        import matplotlib.pyplot as pyplot

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create an Axes object if None provided:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(not ax):
            fig, ax = pyplot.subplots()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare data series to be plotted:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Fseries     = self.numF/self.numNodes if plot_percentages else self.numF
        Eseries     = self.numE/self.numNodes if plot_percentages else self.numE * 0.01
        Dseries     = (self.numD_E+self.numD_I)/self.numNodes if plot_percentages else (self.numD_E+self.numD_I) * 0.01
        D_Eseries   = self.numD_E/self.numNodes if plot_percentages else self.numD_E * 0.01
        D_Iseries   = self.numD_I/self.numNodes if plot_percentages else self.numD_I * 0.01
        Iseries     = self.numI/self.numNodes if plot_percentages else self.numI * 0.01
        Rseries     = self.numR/self.numNodes if plot_percentages else self.numR * 0.01
        Sseries     = self.numS/self.numNodes if plot_percentages else self.numS * 0.01 

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the reference data:      
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(dashed_reference_results):
            dashedReference_tseries  = dashed_reference_results.tseries[::int(self.numNodes/100)]
            # dashedReference_IDEstack = (dashed_reference_results.numI + dashed_reference_results.numD_I + dashed_reference_results.numD_E + dashed_reference_results.numE)[::int(self.numNodes/100)] / (self.numNodes if plot_percentages else 1)
            ax.plot(dashedReference_tseries, dashed_reference_results.numI[::int(self.numNodes/100)], color='#E0E0E0', linestyle='--', label='$I+D+E$ ('+dashed_reference_label+')', zorder=0)
            # ax.plot(dashedReference_tseries, dashedReference_IDEstack, color='#E0E0E0', linestyle='--', label='$I+D+E$ ('+dashed_reference_label+')', zorder=0)
        if(shaded_reference_results):
            shadedReference_tseries  = shaded_reference_results.tseries
            shadedReference_IDEstack = (shaded_reference_results.numI + shaded_reference_results.numD_I + shaded_reference_results.numD_E + shaded_reference_results.numE) / (self.numNodes if plot_percentages else 1)
            ax.fill_between(shaded_reference_results.tseries, shadedReference_IDEstack, 0, color='#EFEFEF', label='$I+D+E$ (Unmitigated)', zorder=0)
            ax.plot(shaded_reference_results.tseries, shadedReference_IDEstack, color='#E0E0E0', zorder=1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the stacked variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        topstack = np.zeros_like(self.tseries)
        if(any(Fseries) and plot_F=='stacked'):
            ax.fill_between(np.ma.masked_where(Fseries<=0, self.tseries), np.ma.masked_where(Fseries<=0, topstack+Fseries), topstack, color=color_F, alpha=0.5, label='$F$', zorder=2)
            ax.plot(        np.ma.masked_where(Fseries<=0, self.tseries), np.ma.masked_where(Fseries<=0, topstack+Fseries),           color=color_F, zorder=3)
            topstack = topstack+Fseries
        if(any(Eseries) and plot_E=='stacked'):
            ax.fill_between(np.ma.masked_where(Eseries<=0, self.tseries), np.ma.masked_where(Eseries<=0, topstack+Eseries), topstack, color=color_E, alpha=0.5, label='Exposed', zorder=2)
            ax.plot(        np.ma.masked_where(Eseries<=0, self.tseries), np.ma.masked_where(Eseries<=0, topstack+Eseries),           color=color_E, zorder=3)
            topstack = topstack+Eseries
        if(combine_D and plot_D_E=='stacked' and plot_D_I=='stacked'):
            ax.fill_between(np.ma.masked_where(Dseries<=0, self.tseries), np.ma.masked_where(Dseries<=0, topstack+Dseries), topstack, color=color_D_E, alpha=0.5, label='Detected', zorder=2)
            ax.plot(        np.ma.masked_where(Dseries<=0, self.tseries), np.ma.masked_where(Dseries<=0, topstack+Dseries),           color=color_D_E, zorder=3)
            topstack = topstack+Dseries
        else:
            if(any(D_Eseries) and plot_D_E=='stacked'):
                ax.fill_between(np.ma.masked_where(D_Eseries<=0, self.tseries), np.ma.masked_where(D_Eseries<=0, topstack+D_Eseries), topstack, color=color_D_E, alpha=0.5, label='$D_E$', zorder=2)
                ax.plot(        np.ma.masked_where(D_Eseries<=0, self.tseries), np.ma.masked_where(D_Eseries<=0, topstack+D_Eseries),           color=color_D_E, zorder=3)
                topstack = topstack+D_Eseries
            if(any(D_Iseries) and plot_D_I=='stacked'):
                ax.fill_between(np.ma.masked_where(D_Iseries<=0, self.tseries), np.ma.masked_where(D_Iseries<=0, topstack+D_Iseries), topstack, color=color_D_I, alpha=0.5, label='$D_I$', zorder=2)
                ax.plot(        np.ma.masked_where(D_Iseries<=0, self.tseries), np.ma.masked_where(D_Iseries<=0, topstack+D_Iseries),           color=color_D_I, zorder=3)
                topstack = topstack+D_Iseries
        if(any(Iseries) and plot_I=='stacked'):
            ax.fill_between(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, topstack+Iseries), topstack, color=color_I, alpha=0.5, label='Infected', zorder=2)
            ax.plot(        np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, topstack+Iseries),           color=color_I, zorder=3)
            topstack = topstack+Iseries
        if(any(Rseries) and plot_R=='stacked'):
            ax.fill_between(np.ma.masked_where(Rseries<=0, self.tseries), np.ma.masked_where(Rseries<=0, topstack+Rseries), topstack, color=color_R, alpha=0.5, label='$R$', zorder=2)
            ax.plot(        np.ma.masked_where(Rseries<=0, self.tseries), np.ma.masked_where(Rseries<=0, topstack+Rseries),           color=color_R, zorder=3)
            topstack = topstack+Rseries
        if(any(Sseries) and plot_S=='stacked'):
            ax.fill_between(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, topstack+Sseries), topstack, color=color_S, alpha=0.5, label='$S$', zorder=2)
            ax.plot(        np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, topstack+Sseries),           color=color_S, zorder=3)
            topstack = topstack+Sseries
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the shaded variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(any(Fseries) and plot_F=='shaded'):
            ax.fill_between(np.ma.masked_where(Fseries<=0, self.tseries), np.ma.masked_where(Fseries<=0, Fseries), 0, color=color_F, alpha=0.5, label='$F$', zorder=4)
            ax.plot(        np.ma.masked_where(Fseries<=0, self.tseries), np.ma.masked_where(Fseries<=0, Fseries),    color=color_F, zorder=5)
        if(any(Eseries) and plot_E=='shaded'):
            ax.fill_between(np.ma.masked_where(Eseries<=0, self.tseries), np.ma.masked_where(Eseries<=0, Eseries), 0, color=color_E, alpha=0.5, label='Exposed', zorder=4)
            ax.plot(        np.ma.masked_where(Eseries<=0, self.tseries), np.ma.masked_where(Eseries<=0, Eseries),    color=color_E, zorder=5)
        if(combine_D and (any(Dseries) and plot_D_E=='shaded' and plot_D_I=='shaded')):
            ax.fill_between(np.ma.masked_where(Dseries<=0, self.tseries), np.ma.masked_where(Dseries<=0, Dseries), 0, color=color_D_E, alpha=0.5, label='Detected', zorder=4)
            ax.plot(        np.ma.masked_where(Dseries<=0, self.tseries), np.ma.masked_where(Dseries<=0, Dseries),    color=color_D_E, zorder=5)
        else:
            if(any(D_Eseries) and plot_D_E=='shaded'):
                ax.fill_between(np.ma.masked_where(D_Eseries<=0, self.tseries), np.ma.masked_where(D_Eseries<=0, D_Eseries), 0, color=color_D_E, alpha=0.5, label='$D_E$', zorder=4)
                ax.plot(        np.ma.masked_where(D_Eseries<=0, self.tseries), np.ma.masked_where(D_Eseries<=0, D_Eseries),    color=color_D_E, zorder=5)
            if(any(D_Iseries) and plot_D_I=='shaded'):
                ax.fill_between(np.ma.masked_where(D_Iseries<=0, self.tseries), np.ma.masked_where(D_Iseries<=0, D_Iseries), 0, color=color_D_I, alpha=0.5, label='$D_I$', zorder=4)
                ax.plot(        np.ma.masked_where(D_Iseries<=0, self.tseries), np.ma.masked_where(D_Iseries<=0, D_Iseries),    color=color_D_I, zorder=5)
        if(any(Iseries) and plot_I=='shaded'):
            ax.fill_between(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, Iseries), 0, color=color_I, alpha=0.5, label='Infected', zorder=4)
            ax.plot(        np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, Iseries),    color=color_I, zorder=5)
        if(any(Sseries) and plot_S=='shaded'):
            ax.fill_between(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, Sseries), 0, color=color_S, alpha=0.5, label='$S$', zorder=4)
            ax.plot(        np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, Sseries),    color=color_S, zorder=5)
        if(any(Rseries) and plot_R=='shaded'):
            ax.fill_between(np.ma.masked_where(Rseries<=0, self.tseries), np.ma.masked_where(Rseries<=0, Rseries), 0, color=color_R, alpha=0.5, label='$R$', zorder=4)
            ax.plot(        np.ma.masked_where(Rseries<=0, self.tseries), np.ma.masked_where(Rseries<=0, Rseries),    color=color_R, zorder=5)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the line variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        if(any(Fseries) and plot_F=='line'):
            ax.plot(np.ma.masked_where(Fseries<=0, self.tseries), np.ma.masked_where(Fseries<=0, Fseries), color=color_F, label='$F$', zorder=6)
        if(any(Eseries) and plot_E=='line'):
            ax.plot(np.ma.masked_where(Eseries<=0, self.tseries), np.ma.masked_where(Eseries<=0, Eseries), color=color_E, label='Exposed', zorder=6)
        if(combine_D and (any(Dseries) and plot_D_E=='line' and plot_D_I=='line')):
            ax.plot(np.ma.masked_where(Dseries<=0, self.tseries), np.ma.masked_where(Dseries<=0, Dseries), color=color_D_E, label='Detected', zorder=6)
        else:
            if(any(D_Eseries) and plot_D_E=='line'):
                ax.plot(np.ma.masked_where(D_Eseries<=0, self.tseries), np.ma.masked_where(D_Eseries<=0, D_Eseries), color=color_D_E, label='$D_E$', zorder=6)
            if(any(D_Iseries) and plot_D_I=='line'):
                ax.plot(np.ma.masked_where(D_Iseries<=0, self.tseries), np.ma.masked_where(D_Iseries<=0, D_Iseries), color=color_D_I, label='$D_I$', zorder=6)
        if(any(Iseries) and plot_I=='line'):
            ax.plot(np.ma.masked_where(Iseries<=0, self.tseries), np.ma.masked_where(Iseries<=0, Iseries), color=color_I, label='Infected', zorder=6)
        if(any(Sseries) and plot_S=='line'):
            ax.plot(np.ma.masked_where(Sseries<=0, self.tseries), np.ma.masked_where(Sseries<=0, Sseries), color=color_S, label='$S$', zorder=6)
        if(any(Rseries) and plot_R=='line'):
            ax.plot(np.ma.masked_where(Rseries<=0, self.tseries), np.ma.masked_where(Rseries<=0, Rseries), color=color_R, label='$R$', zorder=6)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the vertical line annotations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(len(vlines)>0 and len(vline_colors)==0):
            vline_colors = ['gray']*len(vlines)
        if(len(vlines)>0 and len(vline_labels)==0):
            vline_labels = [None]*len(vlines)
        if(len(vlines)>0 and len(vline_styles)==0):
            vline_styles = [':']*len(vlines)
        for vline_x, vline_color, vline_style, vline_label in zip(vlines, vline_colors, vline_styles, vline_labels):
            if(vline_x is not None):
                ax.axvline(x=vline_x, color=vline_color, linestyle=vline_style, alpha=1, label=vline_label)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the plot labels:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_xlabel('Number of Days')
        ax.set_ylabel('Fraction of Population' if plot_percentages else 'number of individuals')
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        ax.set_ylim(0, ylim*100)
        # if(plot_percentages):
        #     ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
        if(legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1], legend_labels[::-1], loc='upper right', facecolor='white', edgecolor='none', framealpha=0.9, prop={'size': 8})
        if(title):
            ax.set_title(title, size=12)
        if(side_title):
            ax.annotate(side_title, (0, 0.5), xytext=(-45, 0), ha='right', va='center',
                size=12, rotation=90, xycoords='axes fraction', textcoords='offset points')
       
        return ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_basic(self, plot_S='line', plot_E='line', plot_I='line',plot_R='line', plot_F='line',
                        plot_D_E='line', plot_D_I='line', combine_D=True,
                        color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                        color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                        dashed_reference_results=None, dashed_reference_label='reference', 
                        shaded_reference_results=None, shaded_reference_label='reference', 
                        vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                        ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                        figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels,
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_infections(self, plot_S=False, plot_E='stacked', plot_I='stacked',plot_R=False, plot_F=False,
                            plot_D_E='stacked', plot_D_I='stacked', combine_D=True,
                            color_S='tab:green', color_E='orange', color_I='crimson', color_R='tab:blue', color_F='black',
                            color_D_E='mediumorchid', color_D_I='mediumorchid', color_reference='#E0E0E0',
                            dashed_reference_results=None, dashed_reference_label='reference', 
                            shaded_reference_results=None, shaded_reference_label='reference', 
                            vlines=[], vline_colors=[], vline_styles=[], vline_labels=[],
                            ylim=None, xlim=None, legend=True, title=None, side_title=None, plot_percentages=True,
                            figsize=(12,8), use_seaborn=True, show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if(use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax, plot_S=plot_S, plot_E=plot_E, plot_I=plot_I,plot_R=plot_R, plot_F=plot_F,
                        plot_D_E=plot_D_E, plot_D_I=plot_D_I, combine_D=combine_D,
                        color_S=color_S, color_E=color_E, color_I=color_I, color_R=color_R, color_F=color_F,
                        color_D_E=color_D_E, color_D_I=color_D_I, color_reference=color_reference,
                        dashed_reference_results=dashed_reference_results, dashed_reference_label=dashed_reference_label, 
                        shaded_reference_results=shaded_reference_results, shaded_reference_label=shaded_reference_label, 
                        vlines=vlines, vline_colors=vline_colors, vline_styles=vline_styles, vline_labels=vline_labels, 
                        ylim=ylim, xlim=xlim, legend=legend, title=title, side_title=side_title, plot_percentages=plot_percentages)

        if(show):
            pyplot.show()

        return fig, ax

def custom_exponential_graph(base_graph=None, scale=100, min_num_edges=0, m=9, n=None):
    # Generate a random preferential attachment power law graph as a starting point.
    # By the way this graph is constructed, it is expected to have 1 connected component.
    # Every node is added along with m=8 edges, so the min degree is m=8.
    if(base_graph):
        graph = base_graph.copy()
    else:
        assert(n is not None), "Argument n (number of nodes) must be provided when no base graph is given."
        graph = networkx.barabasi_albert_graph(n=n, m=m)

    # To get a graph with power-law-esque properties but without the fixed minimum degree,
    # We modify the graph by probabilistically dropping some edges from each node. 
    for node in graph:
        neighbors = list(graph[node].keys())
        quarantineEdgeNum = int( max(min(np.random.exponential(scale=scale, size=1), len(neighbors)), min_num_edges) )
        quarantineKeepNeighbors = np.random.choice(neighbors, size=quarantineEdgeNum, replace=False)
        for neighbor in neighbors:
            if(neighbor not in quarantineKeepNeighbors):
                graph.remove_edge(node, neighbor)
    
    return graph

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def plot_degree_distn(graph, max_degree=None, show=True, use_seaborn=True):
    import matplotlib.pyplot as pyplot
    if(use_seaborn):
        import seaborn
        seaborn.set_style('ticks')
        seaborn.despine()
    # Get a list of the node degrees:
    if type(graph)==np.ndarray:
        nodeDegrees = graph.sum(axis=0).reshape((graph.shape[0],1))   # sums of adj matrix cols
    elif type(graph)==networkx.classes.graph.Graph:
        nodeDegrees = [d[1] for d in graph.degree()]
    else:
        raise BaseException("Input an adjacency matrix or networkx object only.")
    # Calculate the mean degree:
    meanDegree = np.mean(nodeDegrees)
    # Generate a histogram of the node degrees:
    pyplot.hist(nodeDegrees, bins=range(max(nodeDegrees)), alpha=0.5, color='tab:blue', label=('mean degree = %.1f' % meanDegree))
    pyplot.xlim(0, max(nodeDegrees) if not max_degree else max_degree)
    pyplot.xlabel('degree')
    pyplot.ylabel('num nodes')
    pyplot.legend(loc='upper right')
    if(show):
        pyplot.show()