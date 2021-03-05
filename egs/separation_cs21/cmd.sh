# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

export train_cmd="run.pl --mem 2G"
export cuda_cmd="run.pl --mem 2G --gpu 1"
export decode_cmd="run.pl --mem 4G"

#cmdgpu="-p MINI -q GPU -o -l hostname=node23|node24|node25|node26 -eo"
#export cuda_cmd="/share/mini1/sw/mini/jet/latest/tools/submitjob $cmdgpu"

#export train_cmd="queue.pl --mem 2G"
#export decode_cmd="queue.pl --mem 4G"
# the use of cuda_cmd is deprecated, used only in 'nnet1',
#export cuda_cmd="queue.pl --mem 2G --gpu 1"


# NPU setup
#export train_cmd="queue.pl -q all.q --mem 2G"
#export cuda_cmd="/home/work_nfs/common/tools/pyqueue_asr.pl --mem 2G --gpu 1"
#export decode_cmd="/home/work_nfs/common/tools/pyqueue_asr.pl --mem 4G --gpu 1"
#export cuda_cmd="queue.pl --mem 2G --gpu 1 --config conf/gpu.conf"
#export decode_cmd="queue.pl -q all.q --mem 4G"

# Custom setup for MINI grid
#export train_cmd="submitjob -p MINI -q NORMAL -m 2000"
#export cuda_cmd="submitjob -p MINI -q NORMAL -m 2000 -o -l hostname=\"node20|node23|node24|node25\" -eo"
#export decode_cmd="submitjob -p MINI -q NORMAL -m 4000"
