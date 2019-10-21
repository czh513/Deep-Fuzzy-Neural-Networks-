# for some obscure reasons, adding --export=NONE here causes an error
# make sure you don't have offending environment variables before calling this
srun -p gpu_short -t 1:00:00 --pty bash
