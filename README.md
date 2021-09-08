# lcquickfit: rapid transit fitting with eccentricity constraints

## Installing dependencies
- Create new environment and install dependencies with
- *\$ conda env create -f environment.yml*
- *\$ conda activate lcquickfit_env*

## To run code
'duration' mode: fit for eccentricity indirectly and more rapidly by sampling in duration and deriving eccentricity constraints post-sampling
- *\$ python3 lcquickfit.py <TOI number> <output path> 'duration'*
- Example: *\$ python3 lcquickfit.py 1272 '/Users/mason/lcquickfit/' 'duration'* --> Takes ~30 mins to run
  
'full' mode: fit for eccentricity directly by sampling in sqrt(e)cos(omega) & sqrt(e)sin(omega)
- *\$ python3 lcquickfit.py <TOI number> <output path> 'full'*
- Example: *\$ python3 lcquickfit.py 1272 '/Users/mason/lcquickfit/' 'full'* --> Takes ~2-3 hours to run
  
## To run code in segments
- TBA
