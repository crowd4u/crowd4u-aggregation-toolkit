#%%

__all__ = [
    'HSDS_Stan',
    'SeparatedBDS',
]

import os
import cmdstanpy
import arviz as az

from typing import List, Optional
from numpy.typing import NDArray

import attr
import numpy as np
import pandas as pd

from crowdkit.aggregation.classification.majority_vote import MajorityVote
from crowdkit.aggregation.base import BaseClassificationAggregator
from crowdkit.aggregation.utils import get_most_probable_labels, named_series_attrib


class HSDS_Stan():
    r"""
    HSDS_Stan crowd-kit like interface
    Implementation:
        Stan 
        You can choose MCMC or Variational
    """

    def __init__(self, labels, algorithm, *,
                  infer_params=None, init_worker_accuracy=0.75) -> None:
        self.labels = labels    
        self.K = len(labels)
        self.algorithm = algorithm
        self.init_worker_accuracy = init_worker_accuracy
        assert self.algorithm in ["vb","mcmc"]
        if infer_params:
            self.infer_params = infer_params
        else:
            self.infer_params = {}
        self.label2int = {label:i+1 for i,label in enumerate(self.labels)}
        self.K = len(self.label2int.keys()) 
        ## Create model
        model_path = os.path.dirname(__file__) + "/HSDS_stan/step1.stan"
        self.step1_model = cmdstanpy.CmdStanModel(stan_file=model_path)
        model_path = os.path.dirname(__file__) + "/HSDS_stan/step2.stan"
        self.step2_model = cmdstanpy.CmdStanModel(stan_file=model_path)
        self.step1_done = False
        self.step2_done = False 

    def __check_rhat(self, fit, *, step2=False, threshold=1.1):
        print(f"Checking Rhat values")
        idata = az.InferenceData(posterior=fit.draws_xr())
        ## p : class prior
        ## pih : human confusion matrix
        ## pia : ai confusion matrix
        cols  = ["p", "pih", "pia"] if step2 else ["p", "pih"]
        r_hat = az.rhat(idata, var_names=cols)
        convergence = True
        pih_values = r_hat['pih'].values
        p_above = len(r_hat['p'].values[r_hat['p'].values > threshold])
        self.p_unconverged_count = p_above
        print(f"p (r_hat > {threshold}):", p_above)
        if p_above > 0:
            convergence = False
        print(f"pih:")
        self.pih_unconverged_count = []
        for i in range(pih_values.shape[1]):  
            pih_above = len(pih_values[:, i][pih_values[:, i] > threshold])
            self.pih_unconverged_count.append(pih_above)
            print(f"\tClass {i} (r_hat > {threshold}):", pih_above)
            if pih_above > 0:
                convergence = False
        if step2:
            print(f"pia:")
            self.pia_unconverged_count =[]
            pia_values = r_hat['pia'].values
            for i in range(pia_values.shape[1]):
                pia_above = len(pia_values[:, i][pia_values[:, i] > threshold])
                self.pia_unconverged_count.append(pia_above)
                print(f"\tClass {i} (r_hat > {threshold}):", pia_above)
                if pia_above > 0:
                    convergence = False
        return convergence


    def fit_step1(self, hu: pd.DataFrame, check_rhat=True) -> None:
        ## Data Setup
        hu["annoID"] = range(0, len(hu.index))
        self.step1_taskid2int = {task:i+1 for i,task in enumerate(hu['task'].unique())}
        self.step1_human_workerid2int = {worker:i+1 for i,worker in enumerate(hu['worker'].unique())}
        ## Data Transform
        Jh = len(self.step1_human_workerid2int.keys()) # The number of human workers
        Nh = len(hu.index)                             # The number of human annotations
        I = len(self.step1_taskid2int.keys())
        iih = np.zeros(Nh, dtype=int) # the task that the n-th human annotation is for
        jjh = np.zeros(Nh, dtype=int) # the worker who made the n-th human annotation
        yh = np.zeros(Nh, dtype=int)  # the label assigned to the n-th human annotation
        for index, row in hu.iterrows():
            annoID = row["annoID"]
            iih[annoID] = self.step1_taskid2int[row["task"]]
            jjh[annoID] = self.step1_human_workerid2int[row["worker"]]
            yh[annoID] = self.label2int[row["label"]]
        step1_data = {
            "Jh" : Jh,
            "K" : self.K,
            "Nh" : Nh,
            "I" : I,
            "iih" : iih,
            "jjh" : jjh,
            "yh" : yh,
            "r" : self.init_worker_accuracy,
        }
        ## Fitting
        if self.algorithm == "mcmc":
            self.step1_fit = self.step1_model.sample(data=step1_data, output_dir="./outputs/", **self.infer_params)
            if check_rhat:
                self.convergence_step1 = self.__check_rhat(self.step1_fit, step2=False)
                if not self.convergence_step1:
                    print("Warning: Rhat values indicates that the STEP1 model has not converged.")
            raw_step1_p = self.step1_fit.stan_variable("p").mean(axis=0)
            raw_step1_pih = self.step1_fit.stan_variable("pih").mean(axis=0)
        elif self.algorithm == "vb":
            self.step1_fit = self.step1_model.variational(data=step1_data, **self.infer_params)
            raw_step1_p = self.step1_fit.stan_variable("p", mean=True)
            raw_step1_pih = self.step1_fit.stan_variable("pih", mean=True)
        ## Normalize
        ### The Simplex constraint in Stan is very strict, 
        ### so in Step2, we normalize to use as initial values
        self.step1_pih = raw_step1_pih / raw_step1_pih.sum(axis=2, keepdims=True) ## pih[Jh,K,K]
        self.step1_p = raw_step1_p / raw_step1_p.sum(axis=0, keepdims=True) ## p[K]
        self.hu = hu
        self.step1_done = True

    def fit_step2(self, ai : pd.DataFrame, check_rhat=True) -> None:
        assert self.hu is not None
        ai["annoID"] = range(0, len(ai.index))
        concat_df = pd.concat([self.hu, ai])
        self.step2_taskid2int = {task:i+1 for i,task in enumerate(concat_df['task'].unique())}
        self.step2_ai_workerid2int = {worker:i+1 for i,worker in enumerate(ai['worker'].unique())}

        ## Data Transform
        Jh = len(self.step1_human_workerid2int.keys()) # The number of human workers
        Nh = len(self.hu.index)                        # The number of human annotations
        Ja = len(self.step2_ai_workerid2int.keys())    # The number of AI workers
        Na = len(ai.index)                             # The number of AI annotations
        I = len(self.step2_taskid2int.keys())  ## Step2
        iih = np.zeros(Nh, dtype=int) # the task that the n-th human annotation is for  
        jjh = np.zeros(Nh, dtype=int) # the worker who made the n-th human annotation
        yh = np.zeros(Nh, dtype=int)  # the label assigned to the n-th human annotation
        for index, row in self.hu.iterrows():
            annoID = row["annoID"]
            iih[annoID] = self.step2_taskid2int[row["task"]] ## Step2
            jjh[annoID] = self.step1_human_workerid2int[row["worker"]]
            yh[annoID] = self.label2int[row["label"]]
        iia = np.zeros(Na, dtype=int) # the task that the n-th AI annotation is for
        jja = np.zeros(Na, dtype=int) # the worker who made the n-th AI annotation
        ya = np.zeros(Na, dtype=int)  # the label assigned to the n-th AI annotation
        for index, row in ai.iterrows():
            annoID = row["annoID"]
            iia[annoID] = self.step2_taskid2int[row["task"]]
            jja[annoID] = self.step2_ai_workerid2int[row["worker"]]
            ya[annoID] = self.label2int[row["label"]]
        step2_data = {
            "Jh" : Jh,
            "Ja" : Ja,
            "K" : self.K,
            "Nh" : Nh,
            "Na" : Na,
            "I" : I,
            "iih" : iih,
            "jjh" : jjh,
            "yh" : yh,
            "iia" : iia,
            "jja" : jja,
            "ya" : ya,
            "r" : self.init_worker_accuracy,
        }
        if self.step1_done:
            init_data = {
                "p" : self.step1_p,
                "pih" : self.step1_pih,
            }
        # For SeparatedBDS
        # ===========
        else:
            print("Warning: STEP1 has not been executed. Use no init_data.")
            #self.convergence_step1 = True
            init_data = {}
        # ============
        ## Fitting
        if self.algorithm == "mcmc":
            self.step2_fit = self.step2_model.sample(data=step2_data, inits=init_data, output_dir="./outputs/", **self.infer_params)
            if check_rhat:
                self.convergence_step2 = self.__check_rhat(self.step2_fit, step2=True)
                self.convergence = self.convergence_step2
                if not self.convergence:
                    print("Warning: Rhat values indicates that the STEP2 model has not converged")
            self.step2_qz = self.step2_fit.stan_variable("q_z").mean(axis=0)
        elif self.algorithm == "vb":
            self.step2_fit = self.step2_model.variational(data=step2_data, inits=init_data, **self.infer_params)
            self.step2_qz = self.step2_fit.stan_variable("q_z", mean=True)
        self.step2_done = True
        
    def predict(self) -> pd.Series:
        assert self.step2_qz is not None
        predicts = self.step2_qz.argmax(axis=1)
        ## Transform Results
        rows = []
        for k,v in self.step2_taskid2int.items():
            row = {
                "task" : k,
                "label" : self.labels[predicts[v-1]]
            }
            rows.append(row)
        return pd.DataFrame(rows).set_index("task")["label"]


    def fit_predict(self, human: pd.DataFrame, ai: pd.DataFrame, check_rhat=True) -> pd.Series:
        self.fit_step1(human, check_rhat=False)
        # self.fit_step1(human, check_rhat=check_rhat)
        # if check_rhat and not self.convergence_step1:
        #     print("Error!!! Skip STEP2 because STEP1 has not converged.")
        #     return None
        self.fit_step2(ai, check_rhat=check_rhat)
        return self.predict()


        
