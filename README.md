# Variational Inference for the Tobit model

### Scenario 1 (baseline):

n = 2000
d = 200

X_design_sctructure: iid, k = 1, corr = 0

snr = 2
pi0 = 0.05

>burn_in = 500
>n_iters = 1500


#### Scenario 2 (collinearity):
 
n = 2_000
d = 1_000

X_design_structure: corr_blocks, k = 50, corr = 0.9

snr = 1
pi0 = 0.1

>burn_in = 500
>n_iters = 2500

### Scenario 3 (weak signal):

n = 2_000
d = 1_000

X_design_scructure: AR, corr = 0.5

snr = 0.2
pi0 = 0.05

>burn_in = 500
>n_iters = 2000

#### Scenario 4 (d>>n):

n = 5_000
d = 20_000

X_design_sctructure: AR_blocks, k=100, corr = 0.95

snr = 1.0
pi_0 = 0.005

>burn_in = 2000
>n_iters = 4000