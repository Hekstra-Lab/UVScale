from matplotlib import pyplot as plt
import reciprocalspaceship as rs
from argparse import ArgumentParser


desc = """
Make a simple plot of scaling results. 
"""

parser = ArgumentParser(desc)
parser.add_argument('mtz_file', help="An mtz file output from `uvscale`.")
parser = parser.parse_args()

inFN = parser.mtz_file
mtz = rs.read_mtz(inFN).compute_dHKL()

mtz.sort_values('dHKL', inplace=True)                                   
plt.fill_between(mtz.dHKL**-2,                                          
    mtz.SqrtSigma - mtz.SigSqrtSigma,                                   
        mtz.SqrtSigma + mtz.SigSqrtSigma,                                   
            color='b',                                                          
                alpha=0.1,                                                          
                    label="$\sqrt{\Sigma} \pm (std. dev)$"                              
                    )                                                                       
plt.plot(mtz.dHKL**-2, mtz.SqrtSigma, '--r',                            
    label="$\sqrt{\Sigma}$"                                             
    )                                                                       
plt.errorbar(mtz.dHKL**-2, mtz.F, yerr=mtz.SigF, ls='none', color='k',  
    label="$F$"                                                         
    )                                                                       
plt.xlabel("$D_{h}^{-2}\ (\AA^{-2})$")                                  
plt.legend()                                                            
plt.show()
