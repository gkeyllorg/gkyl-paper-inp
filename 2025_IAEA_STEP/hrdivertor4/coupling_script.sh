#!/bin/bash -l

#.Declare a name for this job, preferably with 16 or fewer characters.
#SBATCH -J hrdivertor4
#SBATCH -A m4564

#.Request the queue (enter the possible names, if omitted, default is the default)
#.this job is going to use the default
#SBATCH -q regular

#.Number of nodes to request (Perlmutter has 64 cores and 4 GPUs per node)
#SBATCH -N 2
#SBATCH -n 16

#.Specify GPU needs:
#SBATCH --constraint gpu
#SBATCH --gpus 8

#.Request wall time
#SBATCH -t 24:00:00


#.Mail is sent to you when the job starts and when it terminates or aborts.
#SBATCH --mail-user=akashukla@utexas.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

# Join error and output files in file with the following format
### aSBATCH -i, --input=gdb_1-%j.out 

RUNDIR=$(pwd)
SOLPSRUNDIR=$(pwd)/step_full_device_D+D2_walled_off/gkeyll_w_2D2
GKLIB_PATH=/pscratch/sd/a/akshukla/GK-Neutral_coupling

module load python/3.12
module load PrgEnv-gnu/8.6.0
module load craype-accel-nvidia80
module load cray-mpich/8.1.30
module load cudatoolkit/12.9
module load nccl/2.18.3-cu12
module load cray-libsci/25.09.0
export MPICH_GPU_SUPPORT_ENABLED=0

echo "Loaded modules"

# Find the last frame to restart from
rframe=0
for f in *.gkyl; do
  n=${f##*_}
  n=${n%.gkyl}
  (( n > rframe)) && rframe=$n
done

# Store the bflux data so we can read that before it gets overwritten
cp *_bflux*_${rframe}.gkyl ${RUNDIR}/temp_bflux_data/

# Run the gkyl executable.
echo "Last frame found was $rframe"
echo "Launching Gkeyll..."
rm -f gkeyll_text_output/new_data_flag 
srun -u -n 8 --gpus 8 --exclusive ./hstep27 -g -M -r$rframe &> gkeyll.log &

GKEYLL_PID=$!

echo "Gkeyll PID is "
echo ${GKEYLL_PID}

echo "Starting main loop..."
# Main coordination loop
while kill -0 $GKEYLL_PID 2> /dev/null; do
    if [ -f gkeyll_text_output/new_data_flag ]; then
        frame=$(cat gkeyll_text_output/new_data_flag)
        if [ "${frame}" -eq "${rframe}" ]; then
            echo "Frame is equal to restart frame, copying data"
            mv ${RUNDIR}/temp_bflux_data/* .
        fi
        echo "Detected new Gkeyll output frame ${frame}. Running post-processing..."

        # Run Python post-processing
        python3 ${GKLIB_PATH}/gkeyllCoupling/gkeyllIO/PrepGkeyllData.py config.yaml
        
        echo "Done setting up EIRENE inputs"

        # Store old eirene sources and data
        mkdir ${RUNDIR}/eirene_history/${frame}
        cd ${SOLPSRUNDIR}
        cp input.dat ${RUNDIR}/eirene_history/${frame}/
        mv fort.??? ${RUNDIR}/eirene_history/${frame}/
        cp fort.10 fort.11 fort.13 fort.15 fort.21 fort.22 fort.30 fort.31 fort.33 fort.34 fort.35 fort.44 fort.46 fort.78 fort.85 ${RUNDIR}/eirene_history/${frame}/
        mv run.log.wEir ${RUNDIR}/eirene_history/${frame}/
        cd ${RUNDIR}/

        echo "Done storing EIRENE Data, Running eirene..."

        #Run EIRENE
        tcsh ${SOLPSRUNDIR}/eirene_submission_script

        echo "Done runnine EIRENE, Now converting eirene to Gkeyll input..."

        python3 ${GKLIB_PATH}/gkeyllCoupling/gkeyllIO/PrepEireneData.py config.yaml

        # Remove flags to avoid re-processing
        rm -f gkeyll_text_output/new_data_flag 
        echo "Removed Flag, continuing..."
    fi

    sleep 2  # Tune the polling interval as needed
done

echo "Gkeyll has finished. Pipeline exiting."





