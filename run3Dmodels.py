#!/usr/bin/env python3

import argparse
from dadi import Misc
from dadi import Spectrum
from dadi import Plotting
from dadi import PhiManip
from dadi import Integration
from dadi import Numerics
from dadi import Inference
import os
import numpy


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vcf")
parser.add_argument("-p1", "--population1")
parser.add_argument("-p2", "--population2")
parser.add_argument("-p3", "--population3")
parser.add_argument("-n1", "--projection1", type=int)
parser.add_argument("-n2", "--projection2", type=int)
parser.add_argument("-n3", "--projection3", type=int)
parser.add_argument("-o", "--outputprefix")

args = parser.parse_args()
file = args.vcf
pop1 = args.population1
pop2 = args.population2
pop3 = args.population3
n1 = args.projection1
n2 = args.projection2
n3 = args.projection3
prefix = args.outputprefix

dd = Misc.make_data_dict(file)
joint_fs = Spectrum.from_data_dict(dd , pop_ids =[ pop1, pop2, pop3],projections =[n1 , n2, n3] ,polarized = True )
print(joint_fs.S())

"""

joint_fs = Spectrum.from_data_dict(dd , pop_ids =[ pop1, pop2],projections =[n1 , n2] ,polarized = True )
print(joint_fs.S())

def progression2pop (params,ns,pts):
  Nanc,Np1,Np2,T1,T2 = params
  xx = Numerics.default_grid(pts)
  phi = PhiManip.phi_1D(xx)
  phi = Integration.one_pop(phi, xx, T1, nu=Nanc)
  phi = PhiManip.phi_1D_to_2D(xx, phi)
  phi = Integration.two_pops(phi, xx, T2, nu1=Np1, nu2=Np2)
  fs = Spectrum.from_phi(phi, ns, (xx,xx))
  return fs

params=numpy.array([0.5,0.5,0.5,0.5,0.5]) # population size units have to be multiplied with the ancestral Ne. Divergence times need to be multiplied by 2Ne x generation time
upper_bound=numpy.array([20,20,20,5,5]) # corresponding to upper bound of 20e6 for pop sizes and 1e6 for divergence times
lower_bound=numpy.array([1e-3,1e-3,1e-3,0,0])

ex_progression2pop = Numerics.make_extrap_log_func(progression2pop)

ns=joint_fs.sample_sizes
pts = [40,50]
ll_model_progression2pop_its = []
popt_progression2pop_its = []
n_runs = 1
for i in range(n_runs):
  print(f"Starting run {i+1} out of {n_runs}...")
  p0 = Misc.perturb_params(params, fold=1, upper_bound=upper_bound, lower_bound=lower_bound)
  popt = Inference.optimize_log(p0, joint_fs, ex_progression2pop, pts,lower_bound=lower_bound,upper_bound=upper_bound,verbose=len(params), maxiter=1000)
  model = ex_progression2pop(popt, ns, pts)
  ll_model_progression2pop_its.append(Inference.ll_multinom(model,joint_fs))
  popt_progression2pop_its.append(popt)

numpy.save("_".join(["ll_model_progression2pop_its",prefix]),ll_model_progression2pop_its)
numpy.save("_".join(["popt_progression2pop_its",prefix]), popt_progression2pop_its)



"""

################# scenarios

ns = joint_fs.sample_sizes
pts = [40,50,60]


########### no gene flow

####### split 1

print("running no gene flow split 1\n\n")

def progression_1 (params,ns,pts):
  Nanc,N01,N02,N1,N2,N3,T1,T2,T3 = params
  xx=Numerics.default_grid (pts)
  phi = PhiManip.phi_1D(xx)
  phi = Integration.one_pop(phi, xx, T1, nu=Nanc)
  phi = PhiManip.phi_1D_to_2D(xx, phi)
  phi = Integration.two_pops(phi, xx, T2, nu1=N01, nu2=N02)
  phi = PhiManip.phi_2D_to_3D_split_1(xx, phi)
  phi = Integration.three_pops(phi, xx, T3, nu1=N1, nu2=N2, nu3=N3)
  fs = Spectrum.from_phi(phi, ns, (xx,xx,xx))
  return fs
	
##parameter settings

params=numpy.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]) # population size units have to be multiplied with the ancestral Ne. Divergence times need to be multiplied by 2Ne x generation time
upper_bound=numpy.array([20,20,20,20,20,20,5,5,5]) # corresponding to upper bound of 20e6 for pop sizes and 1e6 for divergence times
lower_bound=numpy.array([1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,0,0,0])

##extrapolate

ex_progression_1 = Numerics.make_extrap_log_func(progression_1)

## iterated optimization to avoid getting stuck in local optima:

ll_model_progression_1_its = []
popt_progression_1_its = []
n_runs = 3
for i in range(n_runs):
  print(f"Starting run {i+1} out of {n_runs}...")
  p0 = Misc.perturb_params(params, fold=1, upper_bound=upper_bound, lower_bound=lower_bound)
  popt = Inference.optimize_log(p0, joint_fs, ex_progression_1, pts,lower_bound=lower_bound,upper_bound=upper_bound,verbose=len(params), maxiter=1000)
  model = ex_progression_1(popt, ns, pts)
  ll_model_progression_1_its.append(Inference.ll_multinom(model,joint_fs))
  popt_progression_1_its.append(popt)

numpy.save("_".join(["ll_model_progression_1_its",prefix]),ll_model_progression_1_its)
numpy.save("_".join(["popt_progression_1_its",prefix]), popt_progression_1_its)


####### split 2

print("running no gene flow split 2\n\n")

def progression_2 (params,ns,pts):
  Nanc,N01,N02,N1,N2,N3,T1,T2,T3 = params
  xx=Numerics.default_grid (pts)
  phi = PhiManip.phi_1D(xx)
  phi = Integration.one_pop(phi, xx, T1, nu=Nanc)
  phi = PhiManip.phi_1D_to_2D(xx, phi)
  phi = Integration.two_pops(phi, xx, T2, nu1=N01, nu2=N02)
  phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)
  phi = Integration.three_pops(phi, xx, T3, nu1=N1, nu2=N2, nu3=N3)
  fs = Spectrum.from_phi(phi, ns, (xx,xx,xx))
  return fs
	
#parameter settings

params=numpy.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]) # population size units have to be multiplied with the ancestral Ne. Divergence times need to be multiplied by 2Ne x generation time
upper_bound=numpy.array([20,20,20,20,20,20,5,5,5]) # corresponding to upper bound of 20e6 for pop sizes and 1e6 for divergence times
lower_bound=numpy.array([1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,0,0,0])

#extrapolate

ex_progression_2 = Numerics.make_extrap_log_func(progression_2)

#iterated optimizations

ll_model_progression_2_its = []
popt_progression_2_its = []
n_runs = 3
for i in range(n_runs):
  print(f"Starting run {i+1} out of {n_runs}...")
  p0 = Misc.perturb_params(params, fold=1, upper_bound=upper_bound, lower_bound=lower_bound)
  popt = Inference.optimize_log(p0, joint_fs, ex_progression_2, pts,lower_bound=lower_bound,upper_bound=upper_bound,verbose=len(params), maxiter=1000)
  model = ex_progression_2(popt, ns, pts)
  ll_model_progression_2_its.append(Inference.ll_multinom(model,joint_fs))
  popt_progression_2_its.append(popt)

numpy.save("_".join(["ll_model_progression_2_its",prefix]),ll_model_progression_2_its)
numpy.save("_".join(["popt_progression_2_its",prefix]), popt_progression_2_its)

########### no gene flow


####### split 1

def geneflow (params,ns,pts):
  Nanc,N01,N1,N02,N2,N3,T1,T2,T3,mp0l2,m12, m23, m13 = params
  xx = Numerics.default_grid(pts)
  phi = PhiManip.phi_1D(xx)
  phi = Integration.one_pop(phi, xx, T1, nu=Nanc)
  phi = PhiManip.phi_1D_to_2D(xx, phi)
  phi = Integration.two_pops(phi, xx, T2, nu1=N01, nu2=N02, m12=m0l2, m21 = m012)
  phi = PhiManip.phi_2D_to_3D_split_1(xx, phi)
  phi = Integration.three_pops(phi, xx, T3, nu1=N1, nu2=N2, nu3=N3, m12=m12, m21=m12, m23=m23, m32=m23, m13=m13, m31=m13)
  fs = Spectrum.from_phi(phi, ns, (xx,xx,xx))
  return fs

  
##parameter settings

params=numpy.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1]) # population size units have to be multiplied with the ancestral Ne. Divergence times need to be multiplied by 2Ne x generation time
upper_bound=numpy.array([20,20,20,20,20,20,5,5,5,20,20,20,20]) # corresponding to upper bound of 20e6 for pop sizes and 1e6 for divergence times
lower_bound=numpy.array([1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,0,0,0,0,0,0,0])


##extrapolate

ex_geneflow = Numerics.make_extrap_log_func(geneflow)

## repeat optimization process multiple times to avoid getting stuck in local optima:

ll_model_geneflow_its = []
popt_geneflow_its = []
n_runs = 3
for i in range(n_runs):
  print(f"Starting run {i+1} out of {n_runs}...")
  p0 = Misc.perturb_params(params, fold=2, upper_bound=upper_bound, lower_bound=lower_bound)
  popt = Inference.optimize_log(p0, joint_fs, ex_geneflow, pts,lower_bound=lower_bound,upper_bound=upper_bound,verbose=len(params), maxiter=1000)
  model = ex_geneflow(popt, ns, pts)
  ll_model_geneflow_its = np.array(ll_model_geneflow_its,Inference.ll_multinom(model,joint_fs))
  popt_geneflow_its = np.array(popt_geneflow_its,popt)


numpy.save("_".join(["ll_model_geneflow_its",prefix]),ll_model_geneflow_its)
numpy.save("_".join(["popt_geneflow_its",prefix]), popt_geneflow_its)
