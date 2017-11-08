PROGRAM exo1_secant_program

USE exo1_secant_function

IMPLICIT NONE

REAL :: a,b,c
INTEGER :: n

print*,"Bonjour. Avant toute chose rappellons que cette méthode n'a d'intérêt que si "
print*,"nous regardons une fonction de classe C1 qui est strictement monotone sur "
print*,"l'intervalle considéré, et surtout qui s'y annule effectivement."
print*,"Sinon, mon résultat sera très probablement faux."

print*,"Donnez-moi une borne inférieure"
read*,a
print*,"Donnez-moi une borne supérieure"
read*,b
print*,"Donnez-moi le nombre d'itérations"
read*,n

c = secant(a,b,n)

print*,"La méthode de la sécante donne l'approximation",c," pour le zéro de f sur l'intervalle [a,b] donné."

END PROGRAM exo1_secant_program
