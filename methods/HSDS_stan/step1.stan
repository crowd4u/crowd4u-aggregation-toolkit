data {
  int<lower=1> I; //number of items
  int<lower=2> K; //number of classes
  
  int<lower=1> Jh; //number of human annotators
  int<lower=1> Nh; //number of human annotations
  int<lower=1,upper=I> iih[Nh]; //the item the n-th human annotation belongs to
  int<lower=1,upper=Jh> jjh[Nh]; //the human annotator which produced the n-th annotation
  int yh[Nh]; //the class of the n-th human annotation

  real<lower=0.0,upper=1.0> r; // Initial Worker Accuracy
}

transformed data {
  vector[K] alpha = rep_vector(1,K); //class prevalence prior
  vector[K] lambda[K]; //annotator abilities prior
  for (k in 1:K) {
    lambda[k] = rep_vector(1,K);
    lambda[k][k] = (r / (1-r)) * (K - 1);
  }
}

parameters {
  simplex[K] p;
  simplex[K] pih[Jh,K];
}

transformed parameters {
  vector[K] log_q_c[I];
  vector[K] log_p;
  vector[K] log_pih[Jh,K];

  log_p = log(p);
  log_pih = log(pih);
  
  for (i in 1:I) 
    log_q_c[i] = log_p;
  
  for (n in 1:Nh) 
    for (h in 1:K)
      log_q_c[iih[n],h] += log_pih[jjh[n],h,yh[n]];
}

model {
  for(j in 1:Jh)
    for(h in 1:K)
      pih[j,h] ~ dirichlet(lambda[h]);
  
  p ~ dirichlet(alpha);
  
  for (i in 1:I)
    target += log_sum_exp(log_q_c[i]);
}

generated quantities {
  vector[K] q_z[I]; //the true class distribution of each item
  
  for(i in 1:I)
    q_z[i] = softmax(log_q_c[i]);
}
