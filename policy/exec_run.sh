#!/bin/bash

###############################################################################
# Code written by Pierre Marza (pierre.marza@insa-lyon.fr)                    #
###############################################################################

singularity exec --nv habitat_sem_exp.sif bash run.sh "$@"
