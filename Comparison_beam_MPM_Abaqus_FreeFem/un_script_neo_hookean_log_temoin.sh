#!/usr/bin/sh
echo "hello world"
#for E in 500 1000 3000
for E in 1500 
do
	for nu in 0.35 0.41 0.45
	do
		if [ $E -eq 500 ]
		then
	    	python simul_beam_MPM_log_temoin_script.py "${E}" "${nu}" 10 
		else
			python simul_beam_MPM_log_temoin_script.py "${E}" "${nu}" 15
		fi  
	done
done


