/* TP.cu */

#include "TP_conf.h"
#include "TP_cpu.h"
#include "TP_kernel.cu"


int main(int argc, char* argv[])
{

  /* ----------------------------------------------------------------

                                   CPU 

     ---------------------------------------------------------------- */


// Message d'erreur s'il manque des paramètres en entrée
  if (argc < 4)
    {
      printf("\nEntrez au moins trois paramètres.\n\n");
      printf("Cet exécutable nécessite les paramètres suivants :\n\n");
      printf("  * 1er  paramètre : fichier de référence.\n");
      printf("  * 2nd  paramètre : fichier requête.\n");
      printf("  * 3ème paramètre : longueur k des k-mots.\n");
      printf("  * 4ème paramètre (optionnel) : longueur de l'affichage des résultats les plus récurrents.\n\n");
      exit(1);
    }

  unsigned long *index_kmers1, *index_kmers2;
  unsigned long i, m;
  int k = atoi(argv[3]);
  unsigned long nb_kmers = power(4, k);
  unsigned long size1 = taille_fichier(argv[1]);
  unsigned long size2 = taille_fichier(argv[2]);
  int top;
  unsigned long valeur = 0;
  char *tab_fichier1, *tab_fichier2;
  char mot[k];
  clock_t initial_time_CPU, initial_time_GPU1, initial_time_GPU2, begin_time;   // Temps initial en micro-secondes
  clock_t final_time_CPU, final_time_GPU1, final_time_GPU2, end_time;           // Temps final   en micro-secondes
  float cpu_time;                                                               // Temps total du CPU en secondes
  float gpu_time1, gpu_time2;                                                   // Temps total du GPU en secondes 
  float total_time;                                                             // Temps total du programme en secondes 

  if (argc == 5)
    top = atoi(argv[4]);
  else
    top = TOP;

  srand(time(NULL));
  begin_time = clock();

  index_kmers1 = (unsigned long*) malloc(nb_kmers*sizeof(unsigned long));
  index_kmers2 = (unsigned long*) malloc(nb_kmers*sizeof(unsigned long));

  for(i=0; i<nb_kmers; i++)
    {
      index_kmers1[i] = 0;
      index_kmers2[i] = 0;
    }

  // Stocke les fichiers rentrés en paramètres dans des tableaux
  stocker(argv[1], size1, tab_fichier1);
  stocker(argv[2], size2, tab_fichier2);

  initial_time_CPU = clock();
  initial_time_GPU1 = clock();

  // Création de l'index de la séquence de référence (donnée en argv[1])
  for (i=0; i<k; i++)
    mot[i] = tab_fichier1[i];

  for (i=k; i<size1; i++)
    {
      valeur = code(mot, k);
      index_kmers1[valeur]++;
      decalage(mot, tab_fichier1[i], k);
    }

  final_time_GPU1 = clock();
  gpu_time1 = (final_time_GPU1 - initial_time_GPU1)*1e-6; // temps du gpu1 (temporaire, se poursuit plus tard)
  
  // Création de l'index de la séquence requête (donnée en argv[2])
  for (i=0; i<k; i++)
    mot[i] = tab_fichier2[i];

  for (i=k; i<size2; i++)
    {
      valeur = code(mot, k);
      index_kmers2[valeur]++;
      decalage(mot, tab_fichier2[i], k);
    }
  free(tab_fichier2);

  // Cherche le nombre de kmers en communs entre les 2 fichiers
  m = find_kmers(index_kmers1, index_kmers2, nb_kmers);

  final_time_CPU = clock();
  cpu_time = (final_time_CPU - initial_time_CPU)*1e-6;


  /* ------------------------- nombre de blocs ----------------------- */

  // Pour la suite, on choisit un nombre de blocs qui fonctionne, par défaut la valeur est 1
  // J'ai trouvé ces valeurs en "testant", sachant qu'il faut suffisament de blocs, mais que l'on dispose d'une limite

  unsigned long NB_BLOCK_SIZE = 1;

  if (k==3)
    NB_BLOCK_SIZE = 16;
  if ((k>3) && (k<8))
    NB_BLOCK_SIZE = 64;
  if (k==8)
    NB_BLOCK_SIZE = 128;
  if (k==9)
    NB_BLOCK_SIZE = 1024;
  if (k>=10)                    
    NB_BLOCK_SIZE = 2048;  // au delà de k=10 inclus, tout est faux, on dépase les limites du nombre de blocs possibles


  /* ----------------------------------------------------------------

                                   GPU 1 :

            version en implémentant find_kmers sur le GPU

     ---------------------------------------------------------------- */


  unsigned long *index_kmers1_device, *index_kmers2_device, *index_kmers2bis_device, *somme_host, *s2;
  dim3 dimBlock(NB_THREAD);

  somme_host = (unsigned long*) malloc(sizeof(unsigned long));
  somme_host[0] = 0;

  cudaMalloc( (void **) &index_kmers1_device, nb_kmers*sizeof(unsigned long) );
  cudaMalloc( (void **) &index_kmers2_device, nb_kmers*sizeof(unsigned long) );
  cudaMalloc( (void **) &index_kmers2bis_device, nb_kmers*sizeof(unsigned long) );
  cudaMalloc( (void **) &s2, nb_kmers/NB_BLOCK_SIZE*sizeof(unsigned long) );

  initial_time_GPU1 = clock();

  // On copie sur le GPU les indexations créées par le CPU
  cudaMemcpy( index_kmers1_device, index_kmers1, nb_kmers*sizeof(unsigned long), cudaMemcpyHostToDevice );
  cudaMemcpy( index_kmers2_device, index_kmers2, nb_kmers*sizeof(unsigned long), cudaMemcpyHostToDevice );
  cudaMemcpy( index_kmers2bis_device, index_kmers2, nb_kmers*sizeof(unsigned long), cudaMemcpyHostToDevice );
  free(index_kmers2);

  // Trouve le nombre de kmers en communs sur le GPU et le stocke dans 'somme_host'
  find_kmers_GPU<<<NB_BLOCK_SIZE, NB_THREAD>>>(index_kmers1_device, index_kmers2_device, s2, nb_kmers, NB_BLOCK_SIZE);
  cudaMemcpy( somme_host, index_kmers2_device, sizeof(unsigned long), cudaMemcpyDeviceToHost );
  cudaFree(index_kmers1_device);
  cudaFree(index_kmers2_device);

  final_time_GPU1 = clock();
  gpu_time1 += (final_time_GPU1 - initial_time_GPU1)*1e-6;

  // Création de l'index sur le GPU
  char *chaine;
  unsigned long *Index_device, *somme_host2, *IndexGPU, *temp_code;
  somme_host2 = (unsigned long*) malloc(sizeof(unsigned long));
  somme_host2[0] = 0;
  IndexGPU = (unsigned long*) malloc(nb_kmers*sizeof(unsigned long));


  /* ----------------------------------------------------------------

                                   GPU 2 :

           version en implémentant find_kmers et en indexant
                la séquence de référence sur le GPU

     ---------------------------------------------------------------- */


  cudaMalloc( (void **) &Index_device, nb_kmers*sizeof(unsigned long) );
  cudaMalloc( (void **) &chaine, size1*sizeof(char) );
  cudaMalloc( (void **) &temp_code, size1*sizeof(unsigned long) );

  initial_time_GPU2 = clock();

  // On copie la séquence de référence sur le GPU puis on crée l'index dessus
  modif_tab(tab_fichier1, size1);
  cudaMemcpy( chaine, tab_fichier1, size1*sizeof(char), cudaMemcpyHostToDevice );
  creation_index_GPU<<<NB_BLOCK_SIZE, NB_THREAD>>>(chaine, temp_code, size1, Index_device, nb_kmers, k);
  cudaFree(temp_code);

  // On récupère sur le CPU l'index du fichier de référence créé sur le GPU 
  cudaMemcpy( IndexGPU, Index_device, nb_kmers*sizeof(unsigned long), cudaMemcpyDeviceToHost );

  // Recherche le nombre de kmers en commun des 2 fichiers sur le GPU
  find_kmers_GPU<<<NB_BLOCK_SIZE, NB_THREAD>>>(Index_device, index_kmers2bis_device, s2, nb_kmers, NB_BLOCK_SIZE);
  cudaMemcpy( somme_host2, index_kmers2bis_device, sizeof(unsigned long), cudaMemcpyDeviceToHost );

  final_time_GPU2 = clock();
  gpu_time2 = (final_time_GPU2 - initial_time_GPU2)*1e-6;

  free(tab_fichier1);
  cudaFree(Index_device);
  cudaFree(chaine);
  cudaFree(s2);
  cudaFree(index_kmers2bis_device);


  /* ----------------------------------------------------------------

                               Résultats

     ---------------------------------------------------------------- */

  end_time = clock();
  total_time = (end_time - begin_time)*1e-6;

  // Calcul et affichage des 'top' k-mers les plus fréquents (top=10 par défaut)
  printf("Les %d %d-mers les plus récurrents dans le fichier de référence sont (d'après le CPU) :\n", top, k);
  les10meilleurs(index_kmers1, nb_kmers, k, top);
//  printf("\nLes %d %d-mers les plus récurrents dans les 2 fichiers sont (d'après le GPU2) :\n", top, k);
//  les10meilleurs(IndexGPU, nb_kmers, k, top);

  printf("\nLes %d %d-mers les moins récurrents dans le fichier de référence sont (d'après le CPU) :\n", top, k);
  les10pires(index_kmers1, nb_kmers, k, top, minimum(size1,size2));

  free(index_kmers1);
  free(IndexGPU);

  // Affichage de données
  printf("\nk........................................... %d\n", k);
  printf("nb_kmers.................................... %d\n", nb_kmers);
  printf("taille de la séquence de référence.......... %ld\n", size1);
  printf("taille de la séquence requête............... %ld\n", size2);
  printf("nombre de blocks utilisés................... %ld\n", NB_BLOCK_SIZE);
  printf("nombre de %d-mers en communs (cpu)........... %ld\n", k, m);
  printf("nombre de %d-mers en communs (gpu)........... %ld\n", k, somme_host[0]);
  printf("nombre de %d-mers en communs (gpu2).......... %ld\n", k, somme_host2[0]);
  printf("indice de correspondance.................... %f\n", (float) m / (float) maximum(size1-k, size2-k));
  printf("temps d'exécution CPU....................... %f s\n", cpu_time);
  printf("temps d'exécution GPU1...................... %f s\n", gpu_time1);
  printf("temps d'exécution GPU2...................... %f s\n\n", gpu_time2);
  printf("temps d'exécution total..................... %f s\n\n", total_time);

  free(somme_host);
  free(somme_host2);

  return 0;
}
