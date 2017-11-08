/* Raulin Alexandre
   Vanuxem Yannick */

#include <stdio.h>
#include <stdlib.h>


/* Fonction qui indente une ligne */

int indent(int x){
  int i;
  for (i=0; i<x; i++){
    printf("    ");
  }
  return 0;
}


/* Fonction Main du programme */

int 
main()
{
  int c;
  enum{DEBUT_LIGNE, NORMAL, DEBUT_COMM, PRE_COMM, COMM, FIN_COMM, CHAINE, MACRO } etat = DEBUT_LIGNE ;
 
  /* Compteur {} */
  int cpt = 0;
  /* Compteur de coherence des commentaires. A la fin du fichier doit valoir 0 */
  int cptComm = 0;
  /* Booleen pour savoir si le nombre et la position des incollades est bien respecte */
  int estCoherent = 1;

  /* Tant qu'on est pas a la fin du fichier */
  while ((c=getchar()) != EOF) {
    switch (etat) {

/* On se trouve en debut de ligne */

    case DEBUT_LIGNE:
      switch (c) {
      case '/':
	etat = DEBUT_COMM;
	break;
      case ' ':
	break;
      case '\t':
	break;
      case '\n':
	break;
      case '{':
	putchar('\n');
	indent(cpt);
	putchar(c);
	putchar('\n');
	cpt++;
	indent(cpt);
	break;
      case '}':
	putchar('\n');
	cpt--;
	indent(cpt);
	putchar(c);
	putchar('\n');
	indent(cpt);
	if (cpt<0) estCoherent=0;
	break;
      case '#':
	putchar(c);
	etat = MACRO;
	break;
      case '"':
	putchar(c);
	etat = CHAINE;
	break;
      default:
	putchar(c);
	etat = NORMAL;
	break;
      }
      break;

/* Etat normal */

    case NORMAL:
      switch (c) {
      case '"':
	putchar(c);
	etat = CHAINE;
	break;
      case '/':
	etat = DEBUT_COMM;
	break;
      case '\n':
	putchar('\n');
	indent(cpt);
	etat = DEBUT_LIGNE;
	break;
      case '{':
	etat = DEBUT_LIGNE;
	putchar('\n');
	indent(cpt);
	putchar(c);
	putchar('\n');
	cpt++;
	indent(cpt);
	break;
      case '}':
	etat = DEBUT_LIGNE;
	putchar('\n');
	cpt--;
	indent(cpt);
	putchar(c);
	putchar('\n');
	indent(cpt);
	if (cpt<0) estCoherent=0;
	break;
      default:
	putchar(c);
	break;
      }
      break;

/* On est au debut d'un possible commentaire */
/*On a lu un '/' mais on l'a pas encore ecrit */

    case DEBUT_COMM:
      switch (c) {
      case '\n':
	putchar('/');
	putchar('\n');
	indent(cpt);
	etat = DEBUT_LIGNE;
	break;
      case '{':
	etat = DEBUT_LIGNE;
        putchar('/');
	putchar('\n');
	indent(cpt);
	putchar(c);
	putchar('\n');
	cpt++;
	indent(cpt);
	break;
      case '}':
	putchar('/');
	etat = DEBUT_LIGNE;
	putchar('\n');
	cpt--;
	indent(cpt);
	putchar(c);
	putchar('\n');
	indent(cpt);
	if (cpt<0) estCoherent=0;
	break;
      case '/':
	putchar(c);
	break;
      case '*':
	putchar('\n');
	etat = PRE_COMM;
	indent(cpt);
	putchar('/');
	putchar('*');
	cptComm++;
	break;
      case '"':
	putchar('/');
	etat = CHAINE;
	putchar('"');
	break;
      default:
	putchar('/');
	putchar(c);
	etat = NORMAL;
	break;
      }
      break;

/* Cet etat gere les espaces au debut les comm */
    case PRE_COMM:
      switch (c) {
      case ' ':
	break;
      case '\t':
	break;
      case '\n':
	break;
      default : 
	putchar(c);
	etat = COMM;
	break;
      }
      break;


 /* On se trouve dans un commentaire */

    case COMM:
      switch (c) {
      case '\n':
	putchar('*');
	putchar('/');
	putchar('\n');
	indent(cpt);
	putchar('/');
	putchar('*');
	etat = PRE_COMM;
	break;
      case '*':
	etat = FIN_COMM;
	break;
      default:
	putchar(c);
	break;
      }
      break;

/* on a lu une * mais pas encore de '/' On est peut etre en fin de commantaire*/

    case FIN_COMM:
      switch (c) {
      case '*':
	putchar('*');
        break;
      case '/':
	etat = DEBUT_LIGNE;
	putchar('*');
	putchar('/');
	putchar('\n');
	indent(cpt);
	cptComm--;
        break;
      default:
	putchar('*');
	etat = COMM;
	putchar(c);
        break;
      }
      break;

/* on a lu '"' On se trouve donc dans une chaine de caractere.*/

    case CHAINE:
      switch (c) {
      case '"':
	putchar('"');
	etat = NORMAL;
	break;
      default:
	putchar(c);
	break;
      }
      break;

/* On a lu '#' on se trouve dans une declaration de macro */

    case MACRO:
      switch (c) {
      case '\n':
	etat = DEBUT_LIGNE;
	putchar('\n');
	indent(cpt);
	break;
      case '/':
	etat = DEBUT_COMM;
	break;
      default:
	putchar(c);
	break;
      }

    }
  }
  if (( (estCoherent == 0) || (cpt !=0 )) && (cptComm != 0)) {
    fprintf(stderr, "Attention Commentaire non ferme ou mal positionne, \n de plus le nombre ou la position des accollades n'est pas coherent \n ");
    exit(EXIT_FAILURE);
  }
  if ( (estCoherent == 0) || (cpt !=0)) {
    fprintf(stderr, "Attention Le nombre ou la position des accollades n'est pas coherent\n");
    exit(EXIT_FAILURE);
  }

  if ( cptComm != 0 ) {
    fprintf(stderr, "Attention Commentaire non ferme ou mal positionne\n");
    exit(EXIT_FAILURE);
  } 

  exit(EXIT_SUCCESS);
}
