# a partir de simul_bayley_2_5_1_oscillations.py
from distutils.log import debug
import numpy as np
import sys
import math

import taichi as ti

ti.init(arch=ti.cpu)  # Try to run on GPU
quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 300000 * quality**2, 128 * quality  #250000
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 0.1e-4 / quality   #1e-4
elapsed_time =ti.field(dtype=float, shape=())
n_substeps =ti.field(dtype=int, shape=())
indice_temoin =ti.field(dtype=int, shape=())

# p_vol, p_rho = (dx )**2, 1   # (dx*0.5)
# p_mass = p_vol * p_rho
#E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio  (original!)
E_top, nu_top = 50000, 0.4  # Young's modulus and Poisson's ratio #e5 #0.1  nu=0.4 0.47
E_bottom, nu_bottom = 500, 0.35  # Young's modulus and Poisson's ratio #e5 #0.1  nu=0.4 0.47
rho_bottom, rho_top = 1, 20

g = 20# gravité 150

if len(sys.argv) > 1:
    n_particles = int(sys.argv[1])
print("n_particles = ", n_particles)
if len(sys.argv) > 2:
    E_top = float(sys.argv[2])
print("E = ", E_top)
if len(sys.argv) > 3:
    nu_top = float(sys.argv[3])
print("nu = ", nu_top) 
if len(sys.argv) == 1:   
    print("pas de parametres")

fichier_res = "MPM_top" + "-E" + str(int(E_top)) + "-nu" + "{:.2f}".format(nu_top) + "-rho" + str(int(rho_top)) + "-g" + str(int(g)) +".res"
fich = open(fichier_res, "w" )
print("writing to ", fichier_res)

# mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
#     (1 + nu) * (1 - 2 * nu))  # Lame parameters
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # deformation gradient
G = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # growth tensor  
St = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # stress tensor                                     
material = ti.field(dtype=int, shape=n_particles)  # material id
object_position = ti.field(dtype=int, shape=n_particles)  # 0 for bottom, 1 for top
object_rho = ti.field(dtype=int, shape=n_particles)
object_E = ti.field(dtype=int, shape=n_particles)
object_nu = ti.field(dtype=float, shape=n_particles)

Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=float,
                         shape=(n_grid, n_grid))  # grid node momentum/velocity

grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass

# geometrical dimensions (L=1 by default)
H_bottom = 0.4; h_top = 0.2  #0.025 0.05
Width_bottom = 0.6 #0.7
Width_top = 0.8
n_particles_top = int(n_particles*h_top/H_bottom)
n_particles_bottom = n_particles - n_particles_top

gamma = 0.99999 #0.9999 #0.98 # damping 0.999 
n_mesh = 15  # size of mesh to select particles where stress directions should be drawn
mesh = ti.field(dtype=int, shape=(n_mesh, n_mesh))   # mesh to draw stress directions
delta_mesh = 1 / n_mesh
to_draw = ti.field(dtype=int, shape=n_particles)  # 1: stress to draw or 0:not draw stress
scale_factor = 0.01 # for stress vectosr

#rgb_list_core = [(0.1, 0.1, 0.1), (0.1, 0.1, 0.2) ,(0.1, 0.1, 0.4) ,(0.1, 0.1, 0.7) ,(0.1, 0.1, 1.0)]
rgb_list_bottom = [(0.1, 0.1, 1.0), (0.1, 0.1, 0.8), (0.1, 0.1, 0.6) ,(0.1, 0.1, 0.4) ,(0.1, 0.1, 0.2) ,(0.1, 0.1, 0.1)]
color_levels_bottom=[ti.rgb_to_hex(rgb) for rgb in rgb_list_bottom]

#rgb_list_cortex = [(0.1, 0.1, 0.1), (0.2, 0.1, 0.1) ,(0.4, 0.1, 0.1) ,(0.7, 0.1, 0.1) ,(1.0, 0.1, 0.1)]
rgb_list_top = [(1.0, 0.1, 0.1), (0.8, 0.1, 0.1), (0.6, 0.1, 0.1) ,(0.4, 0.1, 0.1) ,(0.2, 0.1, 0.1) ,(0.1, 0.1, 0.1)]
color_levels_top=[ti.rgb_to_hex(rgb) for rgb in rgb_list_top]

color_level = ti.field(dtype=int, shape=n_particles)  # color level for heat map of det(F)
#print(color_levels_core)


@ti.func
def compute_to_draw():
    # select particles where stress should be drawn
    for p in x:
        i_float = x[p][0]
        j_float = x[p][1]
        i = int(i_float / delta_mesh)
        j = int(j_float / delta_mesh)
        if mesh[i,j] == 0:
            to_draw[p] = 1
            mesh[i,j] = 1


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        mu_0, lambda_0 = object_E[p] / (2 * (1 + object_nu[p])), object_E[p] * object_nu[p] / ((1 + object_nu[p]) * (1 - 2 * object_nu[p]))  # Lame parameters
        p_vol, p_rho = (dx )**2, object_rho[p]  # (dx*0.5)
        p_mass = p_vol * p_rho

        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) +
                dt * C[p]) @ F[p]  # deformation gradient update            #...EQ. 181 PAGE 42
        h = ti.exp(
            10 *                                                            #...HARDENING=10
            (1.0 -
             Jp[p]))  # Hardening coefficient: snow gets harder when compressed
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        h = 1 # no change
        mu, la = mu_0 * h, lambda_0 * h       
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[
                p] == 0:  # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 2:
            F[p] = U @ sig @ V.transpose(
            )  # Reconstruct elastic deformation gradient after plasticity
        # stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        # ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)                     #...C'EST J*SIGMA..PAGE 20 et 18 et 19

        stress = mu * (F[p] @ F[p].transpose())  + ti.Matrix.identity(float, 2) * (la * ti.log(J) - mu)

        exact_stress = stress/J
        St[p] = exact_stress
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]                                         #...OUI EQ. 29
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)   #...OUI on rajoute dvi à vi calculé avec xp et Cp
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i,
                   j] = (1 / grid_m[i, j]) * grid_v[i,
                                                    j]  # Momentum to velocity
            grid_v[i, j][1] -= dt * g  # gravity 50 10
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            #if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
            if j < 2: 
                grid_v[i, j][0] = 0
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v * gamma   # damping
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)      # equation 3 de supplement article "EQ. 29"
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection
    old_elapsed_time = elapsed_time[None]
    elapsed_time[None] += dt
    n_substeps[None] += 1
    if int(elapsed_time[None]/0.1) > int(old_elapsed_time/0.1) :
        print('elapsed time = ',elapsed_time[None])

#group_size = n_particles // 3
print("total number of particles",n_particles)

@ti.kernel
def initialize():
    for i in range(n_particles):
        if i < n_particles_bottom:
            x[i]=[ti.random()*Width_bottom + (1 - Width_bottom)/2, ti.random()*H_bottom]
            if x[i][0] > 0.49 and x[i][0] < 0.51 and x[i][1] > 0.3 and indice_temoin[None] == 0: 
                indice_temoin[None] = i
            material[i] = 1  # 0: fluid 1: jelly 2: snow
            object_position[i] = 0
            object_E[i]= E_bottom
            object_nu[i] = nu_bottom
            object_rho[i] = rho_bottom
        else:
            x[i]=[ti.random()*Width_top + (1 - Width_top)/2, H_bottom + ti.random()*h_top]
            material[i] = 1  # 0: fluid 1: jelly 2: snow
            object_position[i] = 1          
            object_E[i]= E_top
            object_nu[i] = nu_top
            object_rho[i] = rho_top
        v[i] = ti.Matrix([0, 0])
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
    compute_to_draw()
    elapsed_time[None] = 0.0


initialize()
# x_numpy=x.to_numpy()
# hauteur_moyenne_debut=np.mean(x_numpy[n_particles_bottom:,1]) + h_top/2
# print("hauteur_moyenne_debut = ", hauteur_moyenne_debut)
# hauteur_moyenne_fin = hauteur_moyenne_debut
# hauteur_moyenne_fin_old = hauteur_moyenne_debut
# hauteur_moyenne_en_cours_old = hauteur_moyenne_debut


#gui = ti.GUI("Taichi MLS-MPM-99", res=1024, background_color=0x112F41)

gui = ti.GUI("Taichi MLS-MPM-99", res=1024, background_color= ti.rgb_to_hex([0.1, 0.1, 0.1]))
# liste_hauts = [H_bottom + h_top]
# liste_bas = [0.0]
# compteur_bas = 0
# en_descente = True
liste_hauts = [1.0]
liste_bas = [0.0]
compteur_bas = 0
en_descente = True
y_temoin_old = x[indice_temoin[None]][1]
sortir_boucle_while = False


while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
#while True:
#while elapsed_time[None] < time_limit and (not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT)):    
    loop_range = 30 #int(max(50,2e-3 // dt))
    #print("loop_range :", loop_range)
    for s in range(loop_range):
        substep()
        if n_substeps[None] % 200 == 0 :
            # x_numpy=x.to_numpy()
            # hauteur_moyenne_en_cours=np.mean(x_numpy[n_particles_bottom:,1]) + h_top/2
            # print("hauteur_moyenne_en_cours = ", hauteur_moyenne_en_cours) 
            print("nb_steps = ", n_substeps[None], " indice_temoin =", indice_temoin[None], "particule_temoin =", x[indice_temoin[None]][0], " ",x[indice_temoin[None]][1] )
        y_temoin = x[indice_temoin[None]][1]
        if en_descente:
            if (y_temoin > y_temoin_old):
                liste_bas.append(y_temoin)
                compteur_bas += 1
                print("nb_oscillations = ", compteur_bas )
                en_descente = False
        else:
            if (y_temoin < y_temoin_old):
                liste_hauts.append(y_temoin)
                en_descente = True
        y_temoin_old = y_temoin   

        if compteur_bas > 5 or (elapsed_time[None] > 50):
            delta = liste_hauts[-1] - liste_bas[-1]
            position_equilibre = (liste_hauts[-1] + liste_bas[-1])/2
            if abs(y_temoin - position_equilibre) < 0.005 :
                print("convergence, delta = ", delta)
                sortir_boucle_while = True
                break
    if sortir_boucle_while:
        break


    #x_numpy=x.to_numpy()
    # hauteur_moyenne_en_cours=np.mean(x_numpy[n_particles_bottom:,1]) + h_top/2
    # if en_descente:
    #     if (hauteur_moyenne_en_cours >= hauteur_moyenne_en_cours_old):
    #         liste_bas.append(hauteur_moyenne_en_cours)
    #         compteur_bas += 1
    #         print("nb_oscillations = ", compteur_bas )
    #         en_descente = False
    # else:
    #     if (hauteur_moyenne_en_cours <= hauteur_moyenne_en_cours_old):
    #         liste_hauts.append(hauteur_moyenne_en_cours)
    #         en_descente = True
    # hauteur_moyenne_en_cours_old = hauteur_moyenne_en_cours       


    # F_array = F.to_numpy()[0:n_particles_core]
    # J_array_core = np.array([np.linalg.det(f) for f in F_array])
    # #print(J_array_core[0:50])
    # #hist, bin_edges = np.histogram(J_array_core, bins = len(color_levels_core)+1)
    # # print("hist ", hist)
    # # print("bin_edges ", bin_edges)
    # min_value = np.amin(J_array_core)
    # max_value = np.amax(J_array_core)
    # J_array_core_for_quantile = (J_array_core - min_value)/(max_value - min_value)
    # bin_edges = np.quantile(J_array_core_for_quantile, np.linspace(0, 1, len(color_levels_core)+1)) * (max_value - min_value) + min_value
    # for p in range(n_particles_core):
    #     for i in range(len(color_levels_core)): # -1
    #         if J_array_core[p] >= bin_edges[i] and J_array_core[p] <= bin_edges[i+1]:
    #             color_level[p] = i
    # gui.circles(x.to_numpy()[0:n_particles_core],
    #             radius=1.5,
    #             palette=color_levels_core,
    #             palette_indices=color_level.to_numpy()[0:n_particles_core])


    # F_array = F.to_numpy()[n_particles_core :]
    # J_array_core = np.array([np.linalg.det(f) for f in F_array])
    # min_value = np.amin(J_array_core)
    # max_value = np.amax(J_array_core)    
    # J_array_core_for_quantile = (J_array_core - min_value)/(max_value - min_value)
    # bin_edges = np.quantile(J_array_core_for_quantile, np.linspace(0, 1, len(color_levels_cortex)+1)) * (max_value - min_value) + min_value
    # for p in range(n_particles_core, n_particles):
    #     for i in range(len(color_levels_cortex)):
    #         if J_array_core[p - n_particles_core] >= bin_edges[i] and J_array_core[p - n_particles_core] <= bin_edges[i+1]:
    #             color_level[p] = i   
    # gui.circles(x.to_numpy()[n_particles_core:],
    #         radius=1.5,
    #         palette=color_levels_cortex ,
    #         palette_indices=color_level.to_numpy()[n_particles_core :]) 



    gui.circles(x.to_numpy()[0:n_particles_bottom],
                radius=1.5,
                palette=[0x068587, 0xED553B],
                palette_indices=object_position.to_numpy()[0:n_particles_bottom])
    gui.circles(x.to_numpy()[n_particles_bottom:],
            radius=1.5,
            palette=[0x068587, 0xED553B],
            palette_indices=object_position.to_numpy()[n_particles_bottom:])            

    # # draw stress directions
    # to_draw_array = to_draw.to_numpy()            
    # St_array = St.to_numpy()
    # St_to_draw = St_array[to_draw_array == 1]

    # # print("shape de St_to_draw: ", St_to_draw.shape)
    # # print(St_to_draw[0:5])
    # # print("###########")

    # x_array = x.to_numpy()
    # # print("shape de x_array: ", x_array.shape)
    # # print(x_array[0:5])
    # # print("###########")


    # x_with_stress = x_array[to_draw_array == 1]
    # # print("shape de x_with_stress: ", x_with_stress.shape)
    # # print(x_with_stress[0:5])

    # y_direction = np.full((x_with_stress.shape[0],2),[0.05,0.05])
    # # y_direction[5]=[-0.1,0.1]
    # # print("shape de y_direction: ", y_direction.shape)
    # # print(y_direction[0:5])    
    # # x_a = np.array([[0.1, 0.1], [0.9, 0.1]])
    # # y_a = np.array([[0.3, 0.3], [-0.3, 0.3]])
    # for i in range(x_with_stress.shape[0]):
    #     pass
    #     stress_p = St_to_draw[i]
    #     eigen_val,eigen_vect = np.linalg.eig(stress_p) 

    #     direction = scale_factor * eigen_val[0] * np.linalg.norm(eigen_vect[0]) * eigen_vect[0]
    #     y_direction[i] = direction    #[-0.1, 0.1]
    #     gui.arrows(x_with_stress[i:i+1], y_direction[i:i+1], radius=1, color=0xFFFFFF)

    #     direction = scale_factor * eigen_val[1] * np.linalg.norm(eigen_vect[1]) * eigen_vect[1]
    #     y_direction[i] = direction    #[-0.1, 0.1]
    #     gui.arrows(x_with_stress[i:i+1], y_direction[i:i+1], radius=1, color=0xFFFFFF)
    # #gui.arrows(x_with_stress[0:5], y_direction[0:5], radius=1, color=0xFFFFFF)

    gui.show(
    )  # Change to gui.show(f'{frame:06d}.png') to write images to disk

    # x_numpy=x.to_numpy()
    #hauteur_moyenne_fin=np.mean(x_numpy[n_particles_core:,1]) + small_h/2
    # if (abs(hauteur_moyenne_fin - hauteur_moyenne_fin_old)) < 1e-5 or (elapsed_time > 10):
    #     print ("convergence")
    #     break
    # else:
    #     hauteur_moyenne_fin_old = hauteur_moyenne_fin

    # if math.isnan(hauteur_moyenne_en_cours):
    #     print("hauteur_moyenne_en_cours est Nan")
    #     break

    # if compteur_bas > 5 or (elapsed_time[None] > 5):
    #     delta = liste_hauts[-1] - liste_bas[-1]
    #     hauteur_moyenne_fin = (liste_hauts[-1] + liste_bas[-1])/2
    #     print("convergence, delta = ", delta)
    #     break

scatter_plot_MPM = []
liste_p = []
#x_numpy = x.to_numpy()

#print(x_numpy)
n_pas = 1024
pas = 1/n_pas
for p in range(n_particles_bottom) :
    indice_i = int(x[p][0]/pas)
    liste_p.append([indice_i, x[p][1]])

print("long de liste_p= ", len(liste_p))

for i in range(n_pas):
    ligne_v = [el[1] for el in liste_p if el[0]==i ]
    if len(ligne_v) > 0 :
        min_y = min(ligne_v )
        max_y = max(ligne_v )
        scatter_plot_MPM.extend([[i*pas, min_y], [i*pas, max_y]]) 

print("long de scatter_plot= ", len(scatter_plot_MPM))
for p in scatter_plot_MPM:
    fich.write(str(p[0]) + " " + str(p[1]) + "\n")
  
fich.close()


# max_x = np.amax(x_numpy[:n_particles_bottom,0])
# min_x = np.amin(x_numpy[:n_particles_bottom,0])
# largeur_moyenne_fin = max_x - min_x
# fich.write(str(n_particles/((H_bottom+h_top)*Width)) + " " + str(hauteur_moyenne_fin) + " " 
#     + str(largeur_moyenne_fin) + " " + str(n_substeps[None]) + "\n")
# fich.close()

# print("C'est terminé")
# print("nb d'appels à substep = ", n_substeps[None])
# x_numpy=x.to_numpy()
# print("shape =", x_numpy.shape)
# hauteur_moyenne_fin=np.mean(x_numpy[n_particles_bottom:,1]) + h_top/2
# print("hauteur_moyenne_fin = ", hauteur_moyenne_fin)
# # print("largeur_moyenne_fin = ", largeur_moyenne_fin)

# print("Delta_H_sur_H = ",(hauteur_moyenne_fin - hauteur_moyenne_debut)/hauteur_moyenne_debut)
# print("Calcul théorique = ", -p_rho * g * big_H/(2*E))






