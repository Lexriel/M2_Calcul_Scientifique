menu: fact.o power.o menu.o taylor_exp.o
	gcc menu.o fact.o power.o taylor_exp.o -o menu -Wall

menu.o: menu.c menu.h fact.h power.h taylor_exp.h
	gcc -c menu.c
fact.o: fact.c fact.h
	gcc -c fact.c
power.o: power.c power.h
	gcc -c power.c
taylor_exp.o: taylor_exp.c taylor_exp.h power.h fact.h
	gcc -c taylor_exp.c

clean:
	rm *.o