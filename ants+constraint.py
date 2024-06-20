import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1151)

nb_nodes=17
nb_ants=10
length=2
x_nodes=np.random.random(nb_nodes)*length
y_nodes=np.random.random(nb_nodes)*length
theta=np.linspace(0,2*np.pi,nb_nodes+1)[:-1]
x_nodes=length*np.cos(theta)
y_nodes=length*np.sin(theta)

PH=np.zeros((nb_nodes,nb_nodes),dtype=float) #initialisation des phéromones, PH[i,j] = phéromones entre site i et j
X=np.zeros((nb_ants,nb_nodes+1),dtype=int) #initialisation du tableau tenant compte des positions des fourmis


def show_path(list_of_ants): #fonction qui montre un graphique du chemin emprunté
    for i in list_of_ants:
        plt.plot(x_nodes[X[i,:]],y_nodes[X[i,:]])
    plt.plot(x_nodes,y_nodes,'.')
    plt.show()
    return

def show_pheromones(): #fonction qui montre un graphique des pheromones
    max_PH=np.amax(PH)
    for i in range(nb_nodes):
        for j in range(i+1,nb_nodes):
            if PH[i,j]>=0.2:
                plt.plot([x_nodes[i],x_nodes[j]],[y_nodes[i],y_nodes[j]],c=plt.cm.Greys(PH[i,j]/max_PH))
    plt.show()
    return

def calculate_pheromones(distances): #pheromones = chiffre entre 0 et 1
#ici on estime que la distance moyenne entre deux points est environ la longueur de la boîte/2, on compare donc à ça
    mean_distance=nb_nodes*length/2
    return np.abs(1/(distances-0.65*mean_distance))#(1-np.tanh(-1+distances/(length*(nb_nodes)/2)))/2#ength*nb_nodes-distances#np.exp(-(distances/mean_distance)+1)#

def calculate_distances(X): #calcules les distances totales parcourues par chaque fourmies, retourne un vecteur
    x,y=x_nodes[X],y_nodes[X] #X contient juste les indices, ici on récupère les coordonnées correspondantes
    dist=np.sqrt((x[:,1:]-x[:,:-1])**2+(y[:,1:]-y[:,:-1])**2) #calcul de la distance à chaque pas
    return np.sum(dist,axis=1) #distance totale

def apply_pheromones(X,PH,boost_ant=None): #calcule le delta pheromones et les appliques au tableau de pheromones
    
    dist=np.copy(calculate_distances(X)) #distances
    pheromones=np.copy(calculate_pheromones(dist)) #phéromones

    PH_now=np.zeros_like(PH)
    ant_pheromone=np.zeros_like(PH)
    rank=np.argsort(np.argsort(dist)) #donne le classement de chaque fourmie dans le bon ordre
    mean_PH=np.mean(PH)

    for ant in range(nb_ants):

        if boost_ant==ant: #on donne un avantage à la fourmi qui a donné une nouvelle meilleur solution
            ant_pheromone[X[ant,1:],X[ant,:-1]]=mean_PH*10
        else:
            ant_pheromone[X[ant,1:],X[ant,:-1]]=(1-rank[ant]/nb_ants)*pheromones[ant] #nouvelles pheromones, inclu le classement

        ant_pheromone[X[ant,:-1],X[ant,1:]]=ant_pheromone[X[ant,1:],X[ant,:-1]] #rendre matrice symétrique
        PH_now=PH_now+ant_pheromone
        
    PH=.5*PH+.5*PH_now #faire la moyenne avec avant (évaporation)
    PH_now=None #enlever de la mémoire
    dist=None
    pheromones=None
    
    return PH

weights = np.ones((nb_nodes,nb_nodes),dtype='float')

for i in range(nb_nodes):
    for j in range(nb_nodes):
        if np.abs(i-j)==1:
            weights[i,j]=0.01
        elif np.abs(nb_nodes+i-j)==1:
            weights[i,j]=0.01
        elif np.abs(-nb_nodes+i-j)==1:
            weights[i,j]=0.01

def H(PH): #donne l'«hamiltonien», la distribution de probabilités
    randomness=0.2 #facteur b, combien de chance de base d'aller sur chaque point 
    mean_PH=np.mean(PH)
    background=np.zeros_like(PH)
    for i in range(nb_nodes):
        for j in range(i+1,nb_nodes):
            background[i,j],background[j,i]=randomness*mean_PH,randomness*mean_PH  #on applique la chance de base en utilisant la moyenne de pheromones
            
    fluctuated_PH=(np.copy(PH)+background)*weights
    return normalize_lines(fluctuated_PH) #on normalise

def pick_random_node(current_node,H): #choisis un site aléatoirement selon la distribution H
    if current_node==None:
        return 0
    prob=np.array([np.sum(H[current_node,:i]) for i in range(nb_nodes)]) #créé fonction de repartition selon H
    r=np.random.random() #on choisis un nombre réel entre 0 et 1
    return np.amax((r>=np.array(prob)).nonzero()[0]) #le plus gros chiffre tq r<=prob est choisi

def normalize_lines(array): #fonction qui nomralise tout les lignes d'un tableau à 1
    line_sum = np.sum(array,axis=1) #on trouve la norme de chaque ligne
    return (array.T/line_sum).T #on normalise

#première vague
list_of_nodes=np.arange(nb_nodes) #liste des indices des noeuds

for i in range(nb_ants):
    np.random.shuffle(list_of_nodes) #la trajectoire de chaque fourmi est choisie aléatoirement
    X[i,:]=np.hstack((np.copy(list_of_nodes),list_of_nodes[0]))

PH=apply_pheromones(X,PH)

max_iter=1000 #nombre d'itération maximal
counter=0
criterion=True

distances=calculate_distances(X)
list_fastest_path=[]
fastest_path_length=np.amin(distances)
fastest_path=X[np.argmin(distances),:]

while criterion:
    H0=H(PH)

    for ant in range(nb_ants):
        
        H_ant=np.copy(H0)
        current_node=None
        list_of_nodes=np.arange(nb_nodes)
        
        for i in range(nb_nodes):

            current_node=pick_random_node(current_node,H_ant) #on choisis un pt aléatoire
            H_ant[:,current_node]=0 #on rend la probabilitée de ce point nulle
            H_ant=normalize_lines(H_ant) #on renormalise les probabilitées

            X[ant,i]=current_node #on place la fourmi
        X[ant,-1]=X[ant,0]

    distances=np.copy(calculate_distances(X))
    fastest_path_length_in_iteration=np.amin(distances)
    fastest_ant=np.argmin(distances)
    list_fastest_path.append(fastest_path_length_in_iteration)
    distances=None

    if fastest_path_length_in_iteration<fastest_path_length: #on store le chemin le plus rapide

        fastest_path=np.copy(X[fastest_ant,:])
        fastest_path_length=fastest_path_length_in_iteration

    if counter>=2:
        if list_fastest_path[-1]<list_fastest_path[-2]: #si on a trouvé une nouvelle meilleure solution on lui donne un boost
            PH=apply_pheromones(X,PH,boost_ant=fastest_ant)
    
    else: #sinon on ne fait rien de particulier
        PH=apply_pheromones(X,PH)

    if counter%25==0: #plotting
        max_PH=np.amax(PH)
        for i in range(nb_nodes):
            for j in range(i+1,nb_nodes):
                normalized_PH=PH/np.amax(PH)
                plt.plot([x_nodes[i],x_nodes[j]],[y_nodes[i],y_nodes[j]],c=plt.cm.Greys(normalized_PH[i,j]))
                plt.plot(x_nodes,y_nodes,'.')
        plt.pause(0.1)
        #plt.savefig('pheromones_{}.png'.format(counter))

    max_repeated_iterations=50

    if counter>=max_repeated_iterations:
        criterion=False
        for i in range(len(list_fastest_path)-max_repeated_iterations,len(list_fastest_path)):
            if list_fastest_path[i-1]!=list_fastest_path[i]:
                criterion=True
                break
        if not criterion:
            print('Convergence reached')

    if counter>=max_iter:
        criterion=False
        print('Max iteration reached')
    counter+=1

print(fastest_path)
plt.imshow(H0)
plt.show()

show_pheromones()
plt.plot(x_nodes,y_nodes,'.')
plt.plot(x_nodes[fastest_path],y_nodes[fastest_path])
plt.savefig("adhskjuksatjnka.png")
plt.show()
#plt.savefig('fastest_path.png')