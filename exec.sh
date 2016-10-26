# This points to a file, which should contains hostnames (one per line).
# E.g.,
#
# worker1
# worker2
# worker3
#
MACHINE_CFG=$HOME/machine.cfg
sid=`whoami`

cp $1 /data/opt/tmp/$sid/
cp $3 /data/opt/tmp/$sid/
time pssh -t 0 -P -h ${MACHINE_CFG} -x "-t -t" "ulimit -c unlimited && cd /data/opt/tmp/$sid/ && ls > /dev/null && $@ "
