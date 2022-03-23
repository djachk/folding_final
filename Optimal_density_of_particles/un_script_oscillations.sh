#!/usr/bin/sh
echo "hello world"
fichier="rien"
for nb_particules in 30000 40000 60000 80000 100000 120000 140000 160000 180000 200000 220000 240000 260000 280000 300000 320000 340000 360000
do
    for E in 500 1500 3000
    do
        for nu in 0.35 0.40 0.45
        do
            fichier="particules${nb_particules}E${E}nu${nu}.res"
            touch "$fichier"
            echo "creation de ${fichier}"
            python simul_bayly_2_5_1_oscillations.py "${nb_particules}" "${E}" "${nu}"
            
        done
    done

done
