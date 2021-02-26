sudo cpufreq-set -c 0 --max 2400
taskset -c 0 python speedtest.py # single core speed test
sudo cpufreq-set -c 0 --max 4400
