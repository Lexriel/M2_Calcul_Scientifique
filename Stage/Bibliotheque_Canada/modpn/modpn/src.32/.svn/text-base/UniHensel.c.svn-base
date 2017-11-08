/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
// why need SLP for this? better complexity?

#include "UniHensel.h"


// The invariable to test the interruption business.
extern int Interrupted;

// letters array for print variables.
extern char letters[];



// Inner function for clearing some marks in the nodes.
void
clearTmpMarks(SLG *slg, int markNo){
  int i;
  switch(markNo){
  case 1:
     for(i=0; i<slg->GN; i++) clear_TmpMark1((slg->Nodes)[i]);
     break;
  case 2:
     for(i=0; i<slg->GN; i++) clear_TmpMark2((slg->Nodes)[i]);
     break;
  case 3:
     for(i=0; i<slg->GN; i++) clear_TmpMark3((slg->Nodes)[i]);
     break;
  default:
    printf("other mark-bit hasn't been used (they are reserved).\n");

  }
}


/**
 * copy_operand_insert:
 * @slg: a DAG polynomial.
 * @soltIn: the location (index) to insert a new node into the DAG 'slg'.
 *          We use an array to keeps the nodes of a DGA, so the 'soltIn' 
 *          will be an index for the "nodes array" for hold the new node.  
 * @oper: the new node to be inserted into the DAG.
 * @IDsAry2: auxilious temporary array for testing the if a node has been visited.
 * 
 * Return value: 
 **/


operand
copy_operand_insert(SLG * slg, int * slotIn, operand oper, unsigned short int * IDsAry2){
    int id;
// Change-Code: lift-0
//    operand newoper;
    operand newoper=(operand)my_calloc(1, sizeof(operandObj));
    //    printf("Calling copy_operand_insert()!\n"); fflush(stdout);


     if(is_TmpMark2On(oper)) 
	    { return (slg->Nodes)[IDsAry2[id_of(oper)]];}

	switch (type_of(oper)) {
	case t_sfixn:
	          new_sfixn_ini(newoper, sfixn_val(oper));
                  id = *slotIn;
                  id_set(newoper, id);
                  IDsAry2[id_of(oper)]=id;
                  slg->Nodes[(*slotIn)++]=newoper;
                  set_TmpMark2(oper);
                  return newoper;
	case t_var:
 	         new_var_ini(newoper, var_no(oper));
                 id = *slotIn;
                 id_set(newoper, id);
                  IDsAry2[id_of(oper)]=id;
                 slg->Nodes[(*slotIn)++]=newoper;
                 set_TmpMark2(oper);
                 return newoper;
	case t_varpow:
	         new_varpow_ini(newoper, varPow_no(oper), varPow_e(oper));
                 id = *slotIn;
                 id_set(newoper, id);
                  IDsAry2[id_of(oper)]=id;
                 slg->Nodes[(*slotIn)++]=newoper;
                 set_TmpMark2(oper);
                 return newoper;
	case t_biPlus:
	         new_biPlus_ini(newoper,
                         copy_operand_insert(slg, slotIn, biPlus_oper1(oper), IDsAry2),
			 copy_operand_insert(slg, slotIn, biPlus_oper2(oper), IDsAry2));
                 id = *slotIn;
                 id_set(newoper, id);
                  IDsAry2[id_of(oper)]=id;
                 (slg->Nodes)[(*slotIn)++]=newoper;
                 set_TmpMark2(oper);
                 return newoper;
	case t_biSub:
	         new_biSub_ini(newoper,
                             copy_operand_insert(slg, slotIn, biSub_oper1(oper), IDsAry2),
			     copy_operand_insert(slg, slotIn, biSub_oper2(oper), IDsAry2));
                 id = *slotIn;
                 id_set(newoper, id);
                  IDsAry2[id_of(oper)]=id;
                 (slg->Nodes)[(*slotIn)++]=newoper;
                 set_TmpMark2(oper);
                 return newoper;
	case t_biProd:
	         new_biProd_ini(newoper,
                             copy_operand_insert(slg, slotIn, biProd_oper1(oper), IDsAry2),
			     copy_operand_insert(slg, slotIn, biProd_oper2(oper), IDsAry2));
                 id = *slotIn;
                 id_set(newoper, id);
                  IDsAry2[id_of(oper)]=id;
                 (slg->Nodes)[(*slotIn)++]=newoper;
                 set_TmpMark2(oper);
                 return newoper;
	default:
	  fprintf(stdout,"type %d is not ok.\n",type_of(oper) );
                Throw 112;
		return(NULL);
	}
}







/**
 * copy_operand:
 * @oper: a node.
 * 
 * A shallow copy of a node. 
 * 
 * Return value: 
 **/
static 
operand
copy_operand(operand oper){
// Change-Code: lift-0
//    operand newoper;
    operand newoper=(operand)my_calloc(1, sizeof(operandObj));

	switch (type_of(oper)) {

	case t_poly:
	         new_poly_ini(newoper, poly_poly(oper));
                 return newoper;
	case t_sfixn:
	         new_sfixn_ini(newoper, sfixn_val(oper));
                 return newoper;
	case t_var:
	         new_var_ini(newoper, var_no(oper));
                 return newoper;
	case t_varpow:
	         new_varpow_ini(newoper, varPow_no(oper), varPow_e(oper));
                 return newoper;
	case t_biPlus:
	case t_biSub:
	case t_biProd:
	default:
                
	  fprintf(stdout,"%d operation won't be copied.\n", type_of(oper));
                Throw 113;
		return(NULL);
	}
}







/**
 * alloc_operand:
 * @t: the type of the new node to be created.
 *
 * To allocate a node for a DAG polynomial.
 * 
 * Return value: the newly generated node. 
 **/
static 
operand
alloc_operand(operandType t){
// Change-Code: lift-0
//    operand oper;
    operand oper=(operand)my_calloc(1, sizeof(operandObj));


    // printf("C++ code: operand oper=(operand)calloc(1, sizeof(operandObj));\n");
    //fflush(stdout);


	switch (t) {
	case t_poly:
	         new_poly(oper);
                 return oper;
	case t_sfixn:
	         new_sfixn(oper);
                 return oper;
	case t_var:
	         new_var(oper);
                 return oper;
	case t_varpow:
	         new_varpow(oper);
                 return oper;
	case t_biPlus:
	         new_biPlus(oper);
                 return oper;
	case t_biSub:
	         new_biSub(oper);
                 return oper;
	case t_biProd:
	         new_biProd(oper);
                 return oper;
	default:
	        fprintf(stdout,"No this type.\n");
		return(NULL);
	}
}





// n is the moduli for base ring.
/**
 * random_sfixn:
 * @oper: an empty node.
 * @n: an sfixn value.
 * @id: an node ID.
 * 
 * Generate a new node with the type of "sfixn", and with the value of 'n'. 
 *
 * Return value: 
 **/ 
static 
void
random_sfixn(operand oper, sfixn n, unsigned short int id){
  id_set(oper, id);
(oper->SFIX).sfixnVal=0;
 while(!((oper->SFIX).sfixnVal)) (oper->SFIX).sfixnVal=rand()%n;
}

/**
 * random_var:
 * @oper: an empty node.
 * @N: the index of a variable (X_N).
 * @id: an node ID.
 * 
 * Generate a new node with the type of "variable", and with the value of 'X_N'. 
 *
 * Return value: 
 **/ 
static 
void
random_var(operand oper, sfixn N, unsigned short int id){
  id_set(oper, id);
  (oper->VAR).no=(rand()%N) +1;
}



/**
 * random_varpow:
 * @oper: an empty node.
 * @N: the index of a variable (X_N).
 * @id: an node ID.
 * 
 * Generate a new node with the type of "power of a variable", and with the value of 'X_N ^ 4'. 
 *
 * Return value: 
 **/ 
static 
void
random_varpow(operand oper, sfixn N, unsigned short int id){
  id_set(oper, id);
  // has to be bigger than 2. otherwise it's a var.
  (oper->VARPOW).e=(rand()%4) +2 ;
  (oper->VARPOW).no=(rand()%N) +1;
}


// M is the upper bound of the number of children of a given node.
/**
 * random_biPlus:
 * @Nodes: the nodes vector in the DAG.
 * @oper: an empty node.
 * @S: lower bound of the range to insert the new node.
 * @M: upper bound of the range to insert the new node.
 * @id: the ID of the node.
 * 
 * Generate a node with type of 'binary addition', the operands will be random choosed
 *  from the node array with the range of S..M.
 *
 * Return value: 
 **/
static 
void
random_biPlus(operand *Nodes, operand oper, sfixn S, sfixn M, unsigned short int id){
  sfixn dif=M-S;
  id_set(oper, id);
  (oper->BI_PLUS).oper1=Nodes[(rand()%dif)+S];
  (oper->BI_PLUS).oper2=Nodes[(rand()%dif)+S];
}

/**
 * random_biSub:
 * @Nodes: the nodes vector in the DAG.
 * @oper: an empty node.
 * @S: lower bound of the range to insert the new node.
 * @M: upper bound of the range to insert the new node.
 * @id: the ID of the node.
 * 
 * Generate a node with type of 'binary subtraction', the operands will be 
 * random choosed from the node array with the range of S..M.
 *
 * Return value: 
 **/
static 
void
random_biSub(operand * Nodes, operand oper, sfixn S, sfixn M, unsigned short int id){
  sfixn dif=M-S;
  id_set(oper, id);
  (oper->BI_SUB).oper1=Nodes[(rand()%dif)+S];
  (oper->BI_SUB).oper2=Nodes[(rand()%dif)+S];
}



/**
 * random_biProd:
 * @Nodes: the nodes vector in the DAG.
 * @oper: an empty node.
 * @S: lower bound of the range to insert the new node.
 * @M: upper bound of the range to insert the new node.
 * @id: the ID of the node.
 * 
 * Generate a node with type of 'binary mutiplication', the operands will be 
 * random choosed from the node array with the range of S..M.
 *
 * Return value: 
 **/
static 
void
random_biProd(operand * Nodes, operand oper, sfixn S, sfixn M, unsigned short int id){
  sfixn dif=M-S;
  id_set(oper, id);
  (oper->BI_PROD).oper1=Nodes[(rand()%dif)+S];
  (oper->BI_PROD).oper2=Nodes[(rand()%dif)+S];
}



// suppose slg is already shrinked.
// Just remove the redandunt nodes from the DAG (in-place) operation.
SLG *
removRedun_1(SLG *newslg, SLG * slg){
  //SLG *newslg;
  unsigned short int *IDsAry2=
      (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int));
  // newslg=(SLG *)my_malloc(sizeof(SLG));
  newslg->GN=0;
  //newslg->Nodes=(operand *)my_calloc(slg->GN, sizeof(operand));
  clearTmpMarks(slg, 2);
  //printf("slg->GN = %d.\n", slg->GN);
  //printSLG_Line(slg);
  copy_operand_insert(newslg, &(newslg->GN),(ROOT_G(slg)), IDsAry2);
  //printf("newslg->GN = %d.\n", newslg->GN);
  //shrinkG(newslg, newslg->GN);
  my_free(IDsAry2);
  assert(newslg->GN == slg->GN);
  return newslg;
}





// serve as a copy when no redundants.
// suppose slg is already shrinked.
// Just remove the redandunt nodes from the DAG, return a reduce DAG.
SLG *
removRedun(SLG * slg){
  SLG *newslg;
  unsigned short int *IDsAry2=
      (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int));
  newslg=(SLG *)my_malloc(sizeof(SLG));
  newslg->GN=0;
  newslg->GE=0;
  newslg->Nodes=(operand *)my_calloc(slg->GN, sizeof(operand));
  clearTmpMarks(slg, 2);
  //printf("slg->GN = %d.\n", slg->GN);
  //printSLG_Line(slg);
  copy_operand_insert(newslg, &(newslg->GN),(ROOT_G(slg)), IDsAry2);
  //printf("newslg->GN = %d.\n", newslg->GN);
  shrinkG(newslg, newslg->GN);
  my_free(IDsAry2);
  return newslg;
}




// GN number of Nodes in G.
// n is the modulous of the base ring.
// N is the no of variables.
/**
 * randomSLG:
 * @GN: the number of nodes for the DAG to be generated.
 * @n: the value of "sfixn" nodes for the GAG to be generated .
 * @N: the number of variables for the DAG to be generated..
 * 
 * To generated a random GAG.
 *
 * Return value: the newly generated random GAG.
 **/
SLG *
randomSLG(int GN, sfixn n, sfixn N){
  int j;
  unsigned short int i;
  operandType t;
  SLG *slg;
  sfixn S=GN>>2;
  if(GN<8) Throw 100;
  srand(getSeed());
  slg=(SLG *)my_malloc(sizeof(SLG));
  slg->GN=GN;
  slg->GE=0;
  slg->Nodes=(operand *)my_calloc(GN, sizeof(operand));
  srand(getSeed());
  for(i=0; i<GN; i++){
    if(i<= S ) j=rand()%3; else j=rand()%(Ntypes-4) + 3 ;
    switch(j){
	case 0:
                 t=t_sfixn;
                 (slg->Nodes)[i] = alloc_operand(t);
                 random_sfixn((slg->Nodes)[i], n, i);
                 break;
	case 1:
                 t=t_var;
                 (slg->Nodes)[i] = alloc_operand(t);
                 random_var((slg->Nodes)[i], N, i);
                 break;	  
	case 2:
                 t=t_varpow;
                 (slg->Nodes)[i] = alloc_operand(t);
                 random_varpow((slg->Nodes)[i], N, i);
                 break;	     
	case 3:
                 t=t_biPlus;
                 (slg->Nodes)[i] = alloc_operand(t);
                 random_biPlus(slg->Nodes, (slg->Nodes)[i], i-S,i, i);
                 slg->GE+=2;
                 break;
	case 4:
                 t=t_biSub;
                 (slg->Nodes)[i] = alloc_operand(t);
                 random_biSub(slg->Nodes, (slg->Nodes)[i], i-S,i, i);
                 slg->GE+=2;
                 break;
  	case 5:
                 t=t_biProd;
                 (slg->Nodes)[i] = alloc_operand(t);
                 random_biProd(slg->Nodes, (slg->Nodes)[i], i-S,i, i);
                 slg->GE+=2;
                 break;
	default:
 	         fprintf(stdout,"No this type.\n");
		 return(NULL);
	}


  }
  return slg;
}


/**
 * create_sfixn:
 * @oper: an empty node.
 * @val: the sfixn value for the node to be generated.
 * @id: the ID of the node to be generated.
 * 
 * To create a new node with sfxin-node type, and with sfixn-value "val".
 *
 * Return value: 
 **/
void
create_sfixn(operand oper, sfixn val, unsigned short int id){
  id_set(oper, id);
 (oper->SFIX).sfixnVal=val;
}


// varno is starting from 1
/**
 * create_var:
 * @oper: an empty node.
 * @varno: the index number of the variable of the variable-node to be created.
 * @id: the ID of the node to be generated.
 * 
 * To create a new node with sfixn-node type, and with sfixn-value "val".
 *
 * Return value: 
 **/
static 
void
create_var(operand oper, sfixn varno, unsigned short int id){
  id_set(oper, id);
  (oper->VAR).no=varno;
}


/**
 * create_varpow:
 * @oper: an empty node.
 * @e: an exponeet.
 * @varno: the index number of a variable.
 * @id: the ID for the node to be created.
 * 
 * Return value: 
 **/
static 
void
create_varpow(operand oper, sfixn e, sfixn varno, unsigned short int id){
  id_set(oper, id);
  // has to be bigger than 2. otherwise it's a var.
  (oper->VARPOW).e=e ;
  (oper->VARPOW).no=varno;
}


// M is the upper bound of the number of children of a given node.
/**
 * create_biPlus:
 * @Nodes: the vector keeps the nodes of the DAG.
 * @oper: an empty node.
 * @no1: an index of the nodes-array of the DAG.
 * @no2: an index of the nodes-array of the DAG.
 * @id: the ID of the node.
 * 
 * To create a node with binary-addition type, the left operand will be 
 * nodes-array[no1], and the right operand will be nodes-array[no20.
 *
 * Return value: 
 **/
static 
void
create_biPlus(operand * Nodes, operand oper, sfixn no1, sfixn no2, unsigned short int id){
  id_set(oper, id);
  (oper->BI_PLUS).oper1=Nodes[no1];
  (oper->BI_PLUS).oper2=Nodes[no2];
}


/**
 * create_biSub:
 * @Nodes: the vector keeps the nodes of the DAG.
 * @oper: an empty node.
 * @no1: an index of the nodes-array of the DAG.
 * @no2: an index of the nodes-array of the DAG.
 * @id: the ID of the node.
 * 
 * To create a node with binary-subtraction type, the left operand will be 
 * nodes-array[no1], and the right operand will be nodes-array[no20.
 *
 * Return value: 
 **/
static 
void
create_biSub(operand * Nodes, operand oper, sfixn no1, sfixn no2, unsigned short int id){
  id_set(oper, id);
  (oper->BI_SUB).oper1=Nodes[no1];
  (oper->BI_SUB).oper2=Nodes[no2];
}


/**
 * create_biProd:
 * @Nodes: the vector keeps the nodes of the DAG.
 * @oper: an empty node.
 * @no1: an index of the nodes-array of the DAG.
 * @no2: an index of the nodes-array of the DAG.
 * @id: the ID of the node.
 * 
 * To create a node with binary-multiplication type, the left operand will be 
 * nodes-array[no1], and the right operand will be nodes-array[no20.
 *
 * Return value: 
 **/
static 
void
create_biProd(operand * Nodes, operand oper, sfixn no1, sfixn no2, unsigned short int id){
  id_set(oper, id);
  (oper->BI_PROD).oper1=Nodes[no1];
  (oper->BI_PROD).oper2=Nodes[no2];
}




// F1:=16+10*x1^2+31*x1*x2+51*x1*t+77*x1+95*x2^2+x2*t+x2+55*t^2+28*t
// F2:=43+30*x1^2+27*x1*x2+15*x1*t+59*x1+96*x2^2+72*x2*t+87*x2+47*t^2+90*t
// GN number of Nodes in G.
// n is the modulous of the base ring.
// N is the no of variables
// Generate an fixed example. Used by testing!
SLG *
generateSLG_example_1_F1(){
  int GN = 37;
  operandType t;
  SLG *slg;
 //  sfixn S=GN>>2;
  //if(GN<8) Throw 100;
  //srand(getSeed());
  slg=(SLG *)my_malloc(sizeof(SLG));
  slg->GN=GN;
  slg->GE=0;
  slg->Nodes=(operand *)my_calloc(GN, sizeof(operand));

  t=t_sfixn;

  //[0]
  (slg->Nodes)[0] = alloc_operand(t);
  create_sfixn((slg->Nodes)[0], 16, 0);

  //[1]
  (slg->Nodes)[1] = alloc_operand(t);
  create_sfixn((slg->Nodes)[1], 10, 1);  

  //[2]
  (slg->Nodes)[2] = alloc_operand(t);
  create_sfixn((slg->Nodes)[2], 31, 2);

  //[3]
  (slg->Nodes)[3] = alloc_operand(t);
  create_sfixn((slg->Nodes)[3], 51, 3);

  //[4]
  (slg->Nodes)[4] = alloc_operand(t);
  create_sfixn((slg->Nodes)[4], 77, 4);

  //[5]
  (slg->Nodes)[5] = alloc_operand(t);
  create_sfixn((slg->Nodes)[5], 95, 5);

  //[6]
  (slg->Nodes)[6] = alloc_operand(t);
  create_sfixn((slg->Nodes)[6], 1, 6);

  //[7]
  (slg->Nodes)[7] = alloc_operand(t);
  create_sfixn((slg->Nodes)[7], 1, 7);

  //[8]
  (slg->Nodes)[8] = alloc_operand(t);
  create_sfixn((slg->Nodes)[8], 55, 8);

  //[9]
  (slg->Nodes)[9] = alloc_operand(t);
  create_sfixn((slg->Nodes)[9], 28, 9);


  


  t=t_var;

  //[10]
  (slg->Nodes)[10] = alloc_operand(t);
  create_var((slg->Nodes)[10], 1, 10);

  //[11]
  (slg->Nodes)[11] = alloc_operand(t);
  create_var((slg->Nodes)[11], 2, 11);

  //[12]
  (slg->Nodes)[12] = alloc_operand(t);
  create_var((slg->Nodes)[12], 3, 12);




  t=t_biProd;

  //[13]
  (slg->Nodes)[13] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[13], 11, 11, 13);
  slg->GE+=2;

  //[14]
  (slg->Nodes)[14] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[14], 11, 12, 14);
  slg->GE+=2;

  //[15]
  (slg->Nodes)[15] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[15], 11, 10, 15);
  slg->GE+=2;

  //[16]
  (slg->Nodes)[16] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[16], 12, 12, 16);
  slg->GE+=2;

  //[17]
  (slg->Nodes)[17] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[17], 10, 12, 17);
  slg->GE+=2;

  //[18]
  (slg->Nodes)[18] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[18], 10, 10, 18);
  slg->GE+=2;

  //[19]
  (slg->Nodes)[19] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[19], 1, 13, 19);
  slg->GE+=2;

  //[20]
  (slg->Nodes)[20] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[20], 2, 14, 20);
  slg->GE+=2;

  //[21]
  (slg->Nodes)[21] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[21], 3, 15, 21);
  slg->GE+=2;

  //[22]
  (slg->Nodes)[22] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[22], 4, 11, 22);
  slg->GE+=2;

  //[23]
  (slg->Nodes)[23] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[23], 5, 16, 23);
  slg->GE+=2;

  //[24]
  (slg->Nodes)[24] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[24], 6, 17, 24);
  slg->GE+=2;


  //[25]
  (slg->Nodes)[25] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[25], 7, 12, 25);
  slg->GE+=2;

  //[26]
  (slg->Nodes)[26] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[26], 8, 18, 26);
  slg->GE+=2;

  //[27]
  (slg->Nodes)[27] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[27], 9, 10, 27);
  slg->GE+=2;


                 
  t=t_biPlus;

  //[28]
  (slg->Nodes)[28] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[28], 0, 19, 28);
  slg->GE+=2;
 
  //[29]
  (slg->Nodes)[29] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[29], 28, 20, 29);
  slg->GE+=2;

  //[30]
  (slg->Nodes)[30] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[30], 29, 21, 30);
  slg->GE+=2;

  //[31]
  (slg->Nodes)[31] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[31], 30, 22, 31);
  slg->GE+=2;

  //[32]
  (slg->Nodes)[32] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[32], 31, 23, 32);
  slg->GE+=2;

  //[33]
  (slg->Nodes)[33] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[33], 32, 24, 33);
  slg->GE+=2;

  //[34]
  (slg->Nodes)[34] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[34], 33, 25, 34);
  slg->GE+=2;

  //[35]
  (slg->Nodes)[35] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[35], 34, 26, 35);
  slg->GE+=2;

  //[36]
  (slg->Nodes)[36] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[36], 35, 27, 36);
  slg->GE+=2;


  return slg;
}






// GN number of Nodes in G.
// n is the modulous of the base ring.
// N is the no of variables.
// generate a fix exmaple. Used by testing!
SLG *
generateSLG_example_1_F2(){
  int GN = 37;
  operandType t;
  SLG *slg;
  //  sfixn S=GN>>2;
  //if(GN<8) Throw 100;
  //srand(getSeed());
  slg=(SLG *)my_malloc(sizeof(SLG));
  slg->GN=GN;
  slg->GE=0;
  slg->Nodes=(operand *)my_calloc(GN, sizeof(operand));



  t=t_sfixn;

  //[0]
  (slg->Nodes)[0] = alloc_operand(t);
  create_sfixn((slg->Nodes)[0], 43, 0);

  //[1]
  (slg->Nodes)[1] = alloc_operand(t);
  create_sfixn((slg->Nodes)[1], 30, 1);  

  //[2]
  (slg->Nodes)[2] = alloc_operand(t);
  create_sfixn((slg->Nodes)[2], 27, 2);

  //[3]
  (slg->Nodes)[3] = alloc_operand(t);
  create_sfixn((slg->Nodes)[3], 15, 3);

  //[4]
  (slg->Nodes)[4] = alloc_operand(t);
  create_sfixn((slg->Nodes)[4], 59, 4);

  //[5]
  (slg->Nodes)[5] = alloc_operand(t);
  create_sfixn((slg->Nodes)[5], 96, 5);

  //[6]
  (slg->Nodes)[6] = alloc_operand(t);
  create_sfixn((slg->Nodes)[6], 72, 6);

  //[7]
  (slg->Nodes)[7] = alloc_operand(t);
  create_sfixn((slg->Nodes)[7], 87, 7);

  //[8]
  (slg->Nodes)[8] = alloc_operand(t);
  create_sfixn((slg->Nodes)[8], 47, 8);

  //[9]
  (slg->Nodes)[9] = alloc_operand(t);
  create_sfixn((slg->Nodes)[9], 90, 9);


  


  t=t_var;

  //[10]
  (slg->Nodes)[10] = alloc_operand(t);
  create_var((slg->Nodes)[10], 1, 10);

  //[11]
  (slg->Nodes)[11] = alloc_operand(t);
  create_var((slg->Nodes)[11], 2, 11);

  //[12]
  (slg->Nodes)[12] = alloc_operand(t);
  create_var((slg->Nodes)[12], 3, 12);




  t=t_biProd;

  //[13]
  (slg->Nodes)[13] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[13], 11, 11, 13);
  slg->GE+=2;

  //[14]
  (slg->Nodes)[14] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[14], 11, 12, 14);
  slg->GE+=2;

  //[15]
  (slg->Nodes)[15] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[15], 11, 10, 15);
  slg->GE+=2;

  //[16]
  (slg->Nodes)[16] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[16], 12, 12, 16);
  slg->GE+=2;

  //[17]
  (slg->Nodes)[17] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[17], 10, 12, 17);
  slg->GE+=2;

  //[18]
  (slg->Nodes)[18] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[18], 10, 10, 18);
  slg->GE+=2;

  //[19]
  (slg->Nodes)[19] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[19], 1, 13, 19);
  slg->GE+=2;

  //[20]
  (slg->Nodes)[20] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[20], 2, 14, 20);
  slg->GE+=2;

  //[21]
  (slg->Nodes)[21] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[21], 3, 15, 21);
  slg->GE+=2;

  //[22]
  (slg->Nodes)[22] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[22], 4, 11, 22);
  slg->GE+=2;

  //[23]
  (slg->Nodes)[23] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[23], 5, 16, 23);
  slg->GE+=2;

  //[24]
  (slg->Nodes)[24] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[24], 6, 17, 24);
  slg->GE+=2;


  //[25]
  (slg->Nodes)[25] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[25], 7, 12, 25);
  slg->GE+=2;

  //[26]
  (slg->Nodes)[26] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[26], 8, 18, 26);
  slg->GE+=2;

  //[27]
  (slg->Nodes)[27] = alloc_operand(t);
  create_biProd(slg->Nodes, (slg->Nodes)[27], 9, 10, 27);
  slg->GE+=2;


                 
  t=t_biPlus;

  //[28]
  (slg->Nodes)[28] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[28], 0, 19, 28);
  slg->GE+=2;
 
  //[29]
  (slg->Nodes)[29] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[29], 28, 20, 29);
  slg->GE+=2;

  //[30]
  (slg->Nodes)[30] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[30], 29, 21, 30);
  slg->GE+=2;

  //[31]
  (slg->Nodes)[31] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[31], 30, 22, 31);
  slg->GE+=2;

  //[32]
  (slg->Nodes)[32] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[32], 31, 23, 32);
  slg->GE+=2;

  //[33]
  (slg->Nodes)[33] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[33], 32, 24, 33);
  slg->GE+=2;

  //[34]
  (slg->Nodes)[34] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[34], 33, 25, 34);
  slg->GE+=2;

  //[35]
  (slg->Nodes)[35] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[35], 34, 26, 35);
  slg->GE+=2;

  //[36]
  (slg->Nodes)[36] = alloc_operand(t);
  create_biPlus(slg->Nodes, (slg->Nodes)[36], 35, 27, 36);
  slg->GE+=2;


  return slg;
}






// GN number of Nodes in G.
// To generate a new empty DAG DAG
/**
 * newSLG:
 * @GN: number of nodes.
 * 
 * To generate an empty DAG with "GN" empty nodes.
 *
 * Return value: the newly generated DAG.
 **/
SLG *
newSLG(int GN){
  int i;
  SLG *slg;
  slg=(SLG *)my_malloc(sizeof(SLG));
  slg->GN=GN;
  slg->GE=0;
  slg->Nodes=(operand *)my_calloc(GN, sizeof(operand));
  for(i=0; i<slg->GN; i++) NodeI(slg, i) = NULL;
  return slg;
}



// setup an "dead" mark for a dead node.
void setTmpDeadMark2Operand(operand oper){
	switch (type_of(oper)) {
        case t_poly:
	case t_sfixn:
	case t_var:
	case t_varpow:
	case t_biPlus:
	case t_biSub:
	case t_biProd:
	         set_TmpMark3(oper);
                 break;
	default:
	        fprintf(stdout,"Binary operation won't be my_freed.\n");
                Throw 114;
		break;
	}

}



// A shallow free of a node.
void freeOperand(operand oper){
	switch (type_of(oper)) {

        case t_poly:
	         freePoly(poly_poly(oper));
                 my_free(oper);
                 break;
	case t_sfixn:
	case t_var:
	case t_varpow:
	case t_biPlus:
	case t_biSub:
	case t_biProd:
                 my_free(oper);
                 break;
	default:
	        fprintf(stdout,"Binary operation won't be freed.\n");
                Throw 114;
		break;
	}

}



/**
 * freeSLG:
 * @slg: a DAG.
 * 
 * Deep free the DAG.
 *
 * Return value: 
 **/
void freeSLG(SLG *slg){
  int i, N;
  if(slg){
    if(slg->Nodes){
      N=slg->GN;
      clearTmpMarks(slg, 3);
      for(i=0; i<N; i++){
        if((slg->Nodes)[i]) {
	  if(is_TmpMark3On((slg->Nodes)[i])){
            (slg->Nodes)[i] = NULL;
	  }
	  else{
	    setTmpDeadMark2Operand((slg->Nodes)[i]);
	      }
        }
      }
      for(i=0; i<N; i++){
        if((slg->Nodes)[i]) {
	  freeOperand((slg->Nodes)[i]);
        }
      }   

      my_free(slg->Nodes);
    }
    my_free(slg);
  }   
};







void printOperandG( operand oper){
 
	switch (type_of(oper)) {
	case t_poly:
                 fprintf(stdout, "Node[%d]:= \n", id_of(oper));
                 fflush(stdout);
                 printPoly(poly_poly(oper));      
                 break;
	case t_sfixn:
                 fprintf(stdout, "Node[%d]:= %ld;\n", id_of(oper), ((oper->SFIX).sfixnVal));
                 fflush(stdout);
                 break;
	case t_var:
	         fprintf(stdout, "Node[%d]:= %c;\n", id_of(oper), letters[((oper->VAR).no)]);
                 fflush(stdout);
                 break;
	case t_varpow:
                 fprintf(stdout, "Node[%d]:= %c^%d;\n", id_of(oper), letters[((oper->VARPOW).no)], (oper->VARPOW).e);
                 fflush(stdout);
                 break;
	case t_biPlus:
                 fprintf(stdout, "Node[%d]:= Node[%d] + Node[%d];\n", id_of(oper), id_of(((oper->BI_PLUS).oper1)), id_of(((oper->BI_PLUS).oper2)) );
                  fflush(stdout);
                break;
	case t_biSub:
                 fprintf(stdout, "Node[%d]:= Node[%d] - Node[%d];\n", id_of(oper), id_of(((oper->BI_SUB).oper1)), id_of(((oper->BI_SUB).oper2)) );
                 fflush(stdout); 
                break;
	case t_biProd:
                 fprintf(stdout, "Node[%d]:= Node[%d] *  Node[%d];\n", id_of(oper), id_of(((oper->BI_PROD).oper1)), id_of(((oper->BI_PROD).oper2)) );
                 fflush(stdout);
                 break;
	default:
	        fprintf(stdout,"Node[%d]:= No this type!;\n", id_of(oper));
                 fflush(stdout);
		return;
	}
}




/**
 * printOperand:
 * @oper: an node.
 * 
 * to print the contend of a node.
 *
 * Return value: 
 **/
void printOperand( operand oper){
 
	switch (type_of(oper)) {
	case t_poly:
	  //printf("it's a poly id is %d, address is %d.\n", (int)(ulongfixnum)(id_of(oper)), (int)(ulongfixnum)(poly_poly(oper)));
	  //     fflush(stdout);
                 printPoly(poly_poly(oper));
                 fflush(stdout);
                 break;
	case t_sfixn:
                 printf("%ld\n",  ((oper->SFIX).sfixnVal));
                 fflush(stdout);
                 break;
	case t_var:
	         printf( "%c\n", letters[((oper->VAR).no)]);
                 fflush(stdout);
                 break;
	case t_varpow:
                 printf( "%c^%d\n", letters[((oper->VARPOW).no)], (oper->VARPOW).e);
                 fflush(stdout);
                 break;
	case t_biPlus:
                 printf("Node[%d] + Node[%d]\n", id_of(((oper->BI_PLUS).oper1)), id_of(((oper->BI_PLUS).oper2)) );
                 fflush(stdout);
                 break;
	case t_biSub:
                 printf("Node[%d] - Node[%d]\n", id_of(((oper->BI_SUB).oper1)), id_of(((oper->BI_SUB).oper2)) );
                 fflush(stdout);
                 break;
	case t_biProd:
                 printf("Node[%d] *  Node[%d]\n", id_of(((oper->BI_PROD).oper1)), id_of(((oper->BI_PROD).oper2)) );
                 fflush(stdout);
                 break;
	default:
	         printf("No this type!\n");
                 fflush(stdout);
		return;
	}
}





/**
 * printSLG_Line:
 * @slg: a GAG.
 *
 * To print a DAG.
 * 
 * Return value: 
 **/
void printSLG_Line(SLG * slg){
  int i, N;
  if(slg){
    N=slg->GN;
    printf("# of nodes is %d.\n", N);
    fprintf(stdout,"----\n");
    for(i=0; i<N; i++){
      printOperandG((slg->Nodes)[i]);      
    }
    fprintf(stdout,"----\n");   
  }else
    {fprintf(stdout,"An empty graph.\n");}
};



void printSLG_Line_to(int n, SLG * slg){
  int i;
  if(slg){
    
    printf("# of nodes is %d.\n", slg->GN);
    fprintf(stdout,"----\n");
    if (n > (slg->GN)) {n=(slg->GN);}
    for(i=0; i<n; i++){
      printOperandG((slg->Nodes)[i]);      
    }
    fprintf(stdout,"----\n");   
  }else
    {fprintf(stdout,"An empty graph.\n");}
};



void fprintSLG_Line_to(FILE *file, int n, SLG * slg){
  int i;
  if(slg){
    
    fprintf(file, "# of nodes is %d.\n", slg->GN);
    fprintf(file,"----\n");
   if (n > (slg->GN)) {n=(slg->GN);}
    for(i=0; i<n; i++){
      fprintOperandG(file, (slg->Nodes)[i]);      
    }
    fprintf(file,"----\n");   
  }else
    {fprintf(file,"An empty graph.\n");}
};






/**
 * fprintOperandG:
 * @file: a hanlde of a file.
 * @slg: a node.
 *
 * To print a node into a file.
 * 
 * Return value: 
 **/
void fprintOperandG( FILE *file, operand oper){
 
	switch (type_of(oper)) {
	case t_poly:
                 fprintf(file, "Node[%d]:= \n", id_of(oper));
                 //fflush(file);
                 fprintPoly(file, poly_poly(oper));      
                 break;
	case t_sfixn:
                 fprintf(file, "Node[%d]:= %ld;\n", id_of(oper), ((oper->SFIX).sfixnVal));
		 // fflush(file);
                 break;
	case t_var:
	         fprintf(file, "Node[%d]:= %c;\n", id_of(oper), letters[((oper->VAR).no)]);
                 //fflush(file);
                 break;
	case t_varpow:
                 fprintf(file, "Node[%d]:= %c^%d;\n", id_of(oper), letters[((oper->VARPOW).no)], (oper->VARPOW).e);
                 //fflush(file);
                 break;
	case t_biPlus:
                 fprintf(file, "Node[%d]:= Node[%d] + Node[%d];\n", id_of(oper), id_of(((oper->BI_PLUS).oper1)), id_of(((oper->BI_PLUS).oper2)) );
		 //fflush(file);
                break;
	case t_biSub:
                 fprintf(file, "Node[%d]:= Node[%d] - Node[%d];\n", id_of(oper), id_of(((oper->BI_SUB).oper1)), id_of(((oper->BI_SUB).oper2)) );
                 //fflush(file); 
                break;
	case t_biProd:
                 fprintf(file, "Node[%d]:= Node[%d] *  Node[%d];\n", id_of(oper), id_of(((oper->BI_PROD).oper1)), id_of(((oper->BI_PROD).oper2)) );
                 //fflush(file);
                 break;
	default:
	        fprintf(file,"Node[%d]:= No this type!;\n", id_of(oper));
		//fflush(file);
		return;
	}
}








void fprintSLG_Line(FILE *file, SLG *slg){
  int i, N;
  if(slg){
    N=slg->GN;
    fprintf(file, "// of nodes is %d.\n", N);
    fprintf(file,"----\n");
    for(i=0; i<N; i++){
      fprintOperandG(file, (slg->Nodes)[i]);
    }
    fprintf(file,"----\n");
  }else
    {fprintf(file,"An empty graph.\n");}
};



// d(oper1)/d(oper2), oper2 has to be a variable.
operand
getDerivOfAndInsert_OLD0(SLG * slg, int * slotIn, operand oper1, operand oper2, unsigned short int * IDsAry, unsigned short int * IDsAry2){
  int e, no, id;


// Change-Code: lift-0
//  operand newoper, newoper1, newoper2;
    operand newoper=(operand)my_calloc(1, sizeof(operandObj));
    operand newoper1=(operand)my_calloc(1, sizeof(operandObj));
    operand newoper2=(operand)my_calloc(1, sizeof(operandObj));

    if(is_TmpMark1On(oper1)){
      return (slg->Nodes)[IDsAry[id_of(oper1)]];
    }
     

  assert(is_var(oper2));

  switch (type_of(oper1)) {
	case t_sfixn:
	     new_sfixn_ini(newoper, 0);
             id = *slotIn;
             id_set(newoper, id);
	     IDsAry[id_of(oper1)]=id;
             slg->Nodes[(*slotIn)++]=newoper;
             set_TmpMark1(oper1);
             return newoper;
	case t_var:
	     if(is_sameVar(oper1, oper2)) { new_sfixn_ini(newoper, 1);}
             else {new_sfixn_ini(newoper, 0);}
             id = *slotIn;
             id_set(newoper, id);
	     IDsAry[id_of(oper1)]=id;
             (slg->Nodes)[(*slotIn)++]=newoper;
	     set_TmpMark1(oper1);
             return newoper;
	case t_varpow:
          printf("haven't been tested yet! Throw 11\n");
          fflush(stdout);
	  if(! is_inSameVar(oper1, oper2)) {new_sfixn_ini(newoper, 0);}
          Throw 11;
          else {
	    if (varPow_e(oper1) == 0) {new_sfixn_ini(newoper, 1);}
	    else {  new_sfixn_ini(newoper1, varPow_e(oper1));
                    id = *slotIn;
                    id_set(newoper1, id);
                    (slg->Nodes)[(*slotIn)++]=newoper1;
                    e = (varPow_e(oper1))-1;
                    no = varPow_no(oper1);
                    new_varpow_ini(newoper2, no , e);
                    id = *slotIn;
                    id_set(newoper2, id);
                    (slg->Nodes)[(*slotIn)++]=newoper2;
                    new_biProd_ini(newoper, newoper1, newoper2);
            }
	  }
             id = *slotIn;
             id_set(newoper, id);
	     IDsAry[id_of(oper1)]=id;
             (slg->Nodes)[(*slotIn)++]=newoper;
	     set_TmpMark1(oper1);
             return newoper;
	case t_biPlus:
	   new_biPlus_ini(newoper,
 	   getDerivOfAndInsert_OLD0(slg, slotIn, biPlus_oper1(oper1),oper2, IDsAry,IDsAry2),
	   getDerivOfAndInsert_OLD0(slg, slotIn, biPlus_oper2(oper1),oper2, IDsAry,IDsAry2));
             id = *slotIn;
             id_set(newoper, id);
	     IDsAry[id_of(oper1)]=id;
             (slg->Nodes)[(*slotIn)++]=newoper;
	     set_TmpMark1(oper1);
             return newoper;
	case t_biSub:
	     new_biSub_ini(newoper,
 	     getDerivOfAndInsert_OLD0(slg, slotIn, biSub_oper1(oper1),
                                 oper2,IDsAry,IDsAry2),
	     getDerivOfAndInsert_OLD0(slg, slotIn, biSub_oper2(oper1),
                                 oper2,IDsAry,IDsAry2));
             id = *slotIn;
             id_set(newoper, id);
	     IDsAry[id_of(oper1)]=id;
             (slg->Nodes)[(*slotIn)++]=newoper;
	     set_TmpMark1(oper1);
             return newoper;
	case t_biProd:
	  if(DEBUG22){
	    // printf("\n\n\n >>>> biprod operations: ");
	    // printOperand(oper1);
	  }
               new_biProd_ini(newoper1,
                              getDerivOfAndInsert_OLD0(slg, slotIn,
					 biProd_oper1(oper1), oper2, IDsAry,IDsAry2),
			      copy_operand_insert(slg, slotIn, biPlus_oper2(oper1),IDsAry2)
                              );
                    id = *slotIn;
                    id_set(newoper1, id);
	            (slg->Nodes)[(*slotIn)++]=newoper1;

 	  if(DEBUG22){
	    // printf("\n\n\n >>>> child-1: ");
	    // printOperand(biPlus_oper1(oper1));
	    // printf("\n\n\n >>>> child-2: ");
	    // printOperand(biPlus_oper2(oper1));
	    // printf("\n\n\n >>>> oper2: ");
	    // printOperand(oper2);
	  }

               new_biProd_ini(newoper2,
                          copy_operand_insert(slg, slotIn, biPlus_oper1(oper1),IDsAry2),
                          getDerivOfAndInsert_OLD0(slg, slotIn,
					biProd_oper2(oper1), oper2, IDsAry,IDsAry2)
			      );
                    id = *slotIn;
                    id_set(newoper2, id);
	            (slg->Nodes)[(*slotIn)++]=newoper2;

	  if(DEBUG22){
	    // printf("\n\n\n  newoper1: ");
	    // printOperand(newoper1);
	  }

	  if(DEBUG22){
	    // printf("\n\n\n  newoper2: ");
	    // printOperand(newoper2);
	  }


	       new_biPlus_ini(newoper, newoper1, newoper2);
               id = *slotIn;
               id_set(newoper, id);
	       IDsAry[id_of(oper1)]=id;
	       (slg->Nodes)[(*slotIn)++]=newoper;
	       set_TmpMark1(oper1);
               //fprintf(stdout,"5 <<\n");
	       return newoper;
	default:
	        printf("default\n");
	        fprintf(stdout,"oper1 is an unknow type in function getDerivOfAndInsert_OLD0()!\n");
                Throw 0;
    }

}




// developer level function
SLG *
shrinkG(SLG *slg, int newGN){
  int i;
  //printf("2.1.1.0\n");
  //fflush(stdout);
  operand * vec = (operand *)my_malloc(newGN*sizeof(operand));
  for(i=0; i<newGN; i++) vec[i] = (slg->Nodes)[i];
  //printf("2.1.1.1\n");
  //fflush(stdout);
  my_free(slg->Nodes);
  //printf("2.1.1.2\n");
  //fflush(stdout);
  slg->Nodes = vec;
  slg->GN = newGN;
  return slg;
}


// obsolete!
SLG *
getDerivOfG_OLD0(SLG *slg, operand oper2){
  SLG *derivSlg;
  // to record the old-slg to new-slg, ID 2 ID relation for derivative.
  unsigned short int *IDsAry=
      (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int));
  // to record the old-slg to new-slg, ID 2 ID relation for copy.
  unsigned short int *IDsAry2=
      (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int));

  int In=0, *slotIn=&In;
  derivSlg = newSLG((slg->GN)<<4);
  clearTmpMarks(slg,1);
  clearTmpMarks(slg,2);
  //printf("slg->GN = %d. \n", slg->GN);


  getDerivOfAndInsert_OLD0(derivSlg, slotIn, ROOT_G(slg), oper2, IDsAry, IDsAry2);
  //printf("differentiate on ");
  //printOperand(oper2);
  my_free(IDsAry); my_free(IDsAry2);
  return shrinkG(derivSlg, In);
}






//===================================================================>
// Begin.  
//===================================================================>



// Obsolete!
operand
getDerivOfAndInsertFromRoot(operand zero, operand one, 
                       SLG * slg, int * slotIn, operand oper1, 
                       operand oper2, unsigned short int * IDsAry, 
                       unsigned short int * IDsAry2){
  int e, no, id;

// Change-Code: lift-0
//  operand newoper, newoper1, newoper2;
    operand newoper=(operand)my_calloc(1, sizeof(operandObj));
    operand newoper1=(operand)my_calloc(1, sizeof(operandObj));
    operand newoper2=(operand)my_calloc(1, sizeof(operandObj));

    if(is_TmpMark1On(oper1)){ 
      return (slg->Nodes)[IDsAry[id_of(oper1)]];  
    } 
     
    // printOperand(zero);

  assert(is_var(oper2));

  switch (type_of(oper1)) {
	case t_sfixn:
	  //new_sfixn_ini(zero, 0);
             id = *slotIn;
             if(! id_of(zero)) id_set(zero, id);
	     IDsAry[id_of(oper1)]=id;
             slg->Nodes[(*slotIn)++]=zero;
             set_TmpMark1(oper1); 
             return zero; 
	case t_var:
	     if(is_sameVar(oper1, oper2)) 
	       { //new_sfixn_ini(one, 1);
                id = *slotIn;
                if(! id_of(one))  id_set(one, id);
	        IDsAry[id_of(oper1)]=id;
                (slg->Nodes)[(*slotIn)++]=one;
	        set_TmpMark1(oper1);
                return one;
                }
             else 
	       { new_sfixn_ini(zero, 0);
                id = *slotIn;
                if(! id_of(zero)) id_set(zero, id);
	        IDsAry[id_of(oper1)]=id;
                (slg->Nodes)[(*slotIn)++]=zero;
	        set_TmpMark1(oper1);
                return zero;}
	case t_varpow:
          
          printf("haven't been tested yet. Throw 12\n");
          fflush(stdout);
          Throw 12;
	  if(! is_inSameVar(oper1, oper2)) {
	    //new_sfixn_ini(zero, 0);
                 id = *slotIn;
                 if(! id_of(zero)) id_set(zero, id);
	         IDsAry[id_of(oper1)]=id;
                 (slg->Nodes)[(*slotIn)++]=zero;
	         set_TmpMark1(oper1);
                 return zero;
            }
          else { 
	    if (varPow_e(oper1) == 0) {
	      //new_sfixn_ini(one, 1);
                  id = *slotIn;
                  if(! id_of(one)) id_set(one, id);
	          IDsAry[id_of(oper1)]=id;
                  (slg->Nodes)[(*slotIn)++]=one;
	          set_TmpMark1(oper1);
                  return one;
               }
	    else {  new_sfixn_ini(newoper1, varPow_e(oper1));
                    id = *slotIn;
                    id_set(newoper1, id);
                    (slg->Nodes)[(*slotIn)++]=newoper1;
                    e = (varPow_e(oper1))-1;
                    no = varPow_no(oper1);
                    new_varpow_ini(newoper2, no , e);
                    id = *slotIn;
                    id_set(newoper2, id);
                    (slg->Nodes)[(*slotIn)++]=newoper2;
                    new_biProd_ini(newoper, newoper1, newoper2);
                    id = *slotIn;
                    id_set(newoper, id);
	            IDsAry[id_of(oper1)]=id;
                    (slg->Nodes)[(*slotIn)++]=newoper;
	            set_TmpMark1(oper1);
                    return newoper;
            }
	  }

	case t_biPlus:
	   new_biPlus_ini(newoper, 
			  getDerivOfAndInsertFromRoot(zero,one,slg, slotIn, 
                          biPlus_oper1(oper1),oper2, IDsAry,IDsAry2), 
			  getDerivOfAndInsertFromRoot(zero,one,slg, slotIn, 
                          biPlus_oper2(oper1),oper2, IDsAry,IDsAry2));
             id = *slotIn;
             id_set(newoper, id);
	     IDsAry[id_of(oper1)]=id;
             (slg->Nodes)[(*slotIn)++]=newoper;
	     set_TmpMark1(oper1);
             return newoper;
	case t_biSub:
	     new_biSub_ini(newoper, 
			   getDerivOfAndInsertFromRoot(zero,one,slg, slotIn, 
                           biSub_oper1(oper1), oper2,IDsAry,IDsAry2), 
			   getDerivOfAndInsertFromRoot(zero,one,slg, 
                                    slotIn, biSub_oper2(oper1), 
                                 oper2,IDsAry,IDsAry2));
             id = *slotIn;
             id_set(newoper, id);
	     IDsAry[id_of(oper1)]=id;
             (slg->Nodes)[(*slotIn)++]=newoper;
	     set_TmpMark1(oper1);
             return newoper;
	case t_biProd:
	  if(DEBUG22){
	    // printf("\n\n\n >>>> biprod operations: ");
	    // printOperand(oper1);
	  }
               new_biProd_ini(newoper1, 
                              getDerivOfAndInsertFromRoot(zero,one, slg, slotIn, 
					 biProd_oper1(oper1), oper2, IDsAry,IDsAry2),
			      copy_operand_insert(slg, slotIn, biPlus_oper2(oper1),IDsAry2)
                              );
                    id = *slotIn;
                    id_set(newoper1, id);
	            (slg->Nodes)[(*slotIn)++]=newoper1;

 	  if(DEBUG22){
	    // printf("\n\n\n >>>> child-1: ");
	    // printOperand(biPlus_oper1(oper1));
	    // printf("\n\n\n >>>> child-2: ");
	    // printOperand(biPlus_oper2(oper1));
	    // printf("\n\n\n >>>> oper2: ");
	    // printOperand(oper2);
	  }         

               new_biProd_ini(newoper2, 
                          copy_operand_insert(slg, slotIn, biPlus_oper1(oper1),IDsAry2),
			      getDerivOfAndInsertFromRoot(zero,one, slg, slotIn, 
					biProd_oper2(oper1), oper2, IDsAry,IDsAry2)
			      );
                    id = *slotIn;
                    id_set(newoper2, id);
	            (slg->Nodes)[(*slotIn)++]=newoper2;

	  if(DEBUG22){
	    // printf("\n\n\n  newoper1: ");
	    // printOperand(newoper1);
	  }

	  if(DEBUG22){
	    // printf("\n\n\n  newoper2: ");
	    // printOperand(newoper2);
	  }


	       new_biPlus_ini(newoper, newoper1, newoper2);
               id = *slotIn;
               id_set(newoper, id);
	       IDsAry[id_of(oper1)]=id;
	       (slg->Nodes)[(*slotIn)++]=newoper;
	       set_TmpMark1(oper1);
               //fprintf(stdout,"5 <<\n");
	       return newoper;	       
	default:
	        printf("default\n");
	        fprintf(stdout,"oper1 is an unknow type in function getDerivOfAndInsertFromRoot()!\n");
                Throw 0;
    }

}







//===================================================>












/**
 * getDerivOfAndInsertFromRoot_Hashing:
 * @slg: a DAG.
 * @slotIn: the starting slot to insert the result.
 * @oper1: the expression to be differetiated.
 * @oper2: at which variable-node the differentiation conducted.
 * @IDsAry: an auxilious array.
 * @IDsAry2: an auxilious array.
 * @hash_table: a hash table.
 * 
 * Return value: 
 **/
operand
getDerivOfAndInsertFromRoot_Hashing(
                       SLG * slg, int * slotIn, operand oper1, 
                       operand oper2, unsigned short int * IDsAry, 
		       unsigned short int * IDsAry2, HASHTABLE *hash_table){
  int e, no, id;

  // Change-Code: lift-0
  //  operand one, zero;
  //  operand newoper, newoper1, newoper2, tmpoper;
  operand one=(operand)my_calloc(1, sizeof(operandObj));
  operand zero=(operand)my_calloc(1, sizeof(operandObj));
  operand newoper=(operand)my_calloc(1, sizeof(operandObj));
  operand newoper1=(operand)my_calloc(1, sizeof(operandObj));
  operand newoper2=(operand)my_calloc(1, sizeof(operandObj));
  operand tmpoper=(operand)my_calloc(1, sizeof(operandObj));

  //printf("var_no = %ld\n", var_no(oper2));



    if(is_TmpMark1On(oper1)){ 
      return (slg->Nodes)[IDsAry[id_of(oper1)]];  
    } 
     
    // printOperand(zero);

  assert(is_var(oper2));

  switch (type_of(oper1)) {
	case t_sfixn:
	  //printf("t_sfix\n");
	     new_sfixn_ini(zero, 0);
             id = *slotIn;

	     //printf("-3  --id = %ld\n", id);

              id_set(zero, id);
	     IDsAry[id_of(oper1)]=id;
             slg->Nodes[(*slotIn)++]=zero;
             set_TmpMark1(oper1); 
             return zero; 
	case t_var:

	  // printf("t_var\n");
          //printf("var_no(oper1)=%ld\n", var_no(oper1));
          //printf("var_no(oper2)=%ld\n", var_no(oper2));
	     if(is_sameVar(oper1, oper2)) 
	       { //printf("the same\n");
                new_sfixn_ini(one, 1);
                id = *slotIn;

		//printf("-1  --id = %ld\n", id);

                id_set(one, id);
	        IDsAry[id_of(oper1)]=id;
                (slg->Nodes)[(*slotIn)++]=one;
	        set_TmpMark1(oper1);
                return one;
                }
             else 
	       { //printf("Not the same\n");
		 new_sfixn_ini(zero, 0);
                id = *slotIn;

                //printf("-2  --id = %ld\n", id);


                id_set(zero, id);
	        IDsAry[id_of(oper1)]=id;
                (slg->Nodes)[(*slotIn)++]=zero;
	        set_TmpMark1(oper1);
                return zero;}
	case t_varpow:

	  printf("t_varpow\n");

          printf("The code for t_varpow case does NOT finished because when do var^e -> (var+y0)^e we need to expand the power, so we got no gain anymore!!! even we keep the node in the format (var+y0)^e, when converting to cube-polynomial mod TriSet, we need to powPoly() which has the same complexity as we expand the var^e or (var+y0)^e at the beginning !!!! TRY to find smart solution!!!!\n");
          fflush(stdout);
          Throw 13;


	  //printf("INPUT: %ld^%ld / %ld\n", varPow_no(oper1), varPow_e(oper1), var_no(oper2));
  

	  //printf("varPow_no(oper1)=%ld\n.", varPow_no(oper1));
	  //printf("var_no(oper2)=%ld\n.", var_no(oper2));


	     if(varPow_no(oper1) == var_no(oper2)) 
	       { 
		 //printf("the same var\n");

		 if(varPow_e(oper1)==0){
		   new_sfixn_ini(zero, 0);
                    id = *slotIn;
                    id_set(zero, id);
	            IDsAry[id_of(oper1)]=id;

                    //printf("1--id = %ld\n", id);

                    (slg->Nodes)[(*slotIn)++]=zero;
	            set_TmpMark1(oper1);
                    //printf("OUTPUT: 0.\n");
                    return zero;
		 }

                 if(varPow_e(oper1)==1){
		   new_sfixn_ini(one, 1);
                    id = *slotIn;
                    id_set(one, id);

                    //printf("2--id = %ld\n", id);


	            IDsAry[id_of(oper1)]=id;
                    (slg->Nodes)[(*slotIn)++]=one;
	            set_TmpMark1(oper1);
                    //printf("OUTPUT: 1.\n");
                    return one;
		 }




	            new_sfixn_ini(newoper1, varPow_e(oper1));
                    id = *slotIn;
                    id_set(newoper1, id);

                    //printf("3--id = %ld\n", id);


                    (slg->Nodes)[(*slotIn)++]=newoper1;
                    e = (varPow_e(oper1))-1;
                    no = varPow_no(oper1);
                    new_varpow_ini(newoper2, no , e);
                    id = *slotIn;
                    id_set(newoper2, id);


                    //printf("4--id = %ld\n", id);

                    (slg->Nodes)[(*slotIn)++]=newoper2;
                    new_biProd_ini(newoper, newoper1, newoper2);
                    id = *slotIn;
                    id_set(newoper, id);

                    //printf("5--id = %ld\n", id);


	            IDsAry[id_of(oper1)]=id;
                    (slg->Nodes)[(*slotIn)++]=newoper;
	            set_TmpMark1(oper1);

                    //printf("OUTPUT: %ld * %ld ^ %ld.\n\n", varPow_e(oper1), no, e);
                    return newoper;

                }
             else 
	       { 

		 //printf("different var\n");

		new_sfixn_ini(zero, 0);
                id = *slotIn;
                id_set(zero, id);


                //printf("5--id = %ld\n", id);

	        IDsAry[id_of(oper1)]=id;
                (slg->Nodes)[(*slotIn)++]=zero;
	        set_TmpMark1(oper1);
		//printf("OUTPUT: 0.\n");
                return zero;}








/* 	  if((! is_inSameVar(oper1, oper2)) || varPow_e(oper1) == 0) { */
/* 	    //new_sfixn_ini(zero, 0); */
/*                  id = *slotIn; */
/*                  if(! id_of(zero)) id_set(zero, id); */
/* 	         IDsAry[id_of(oper1)]=id; */
/*                  (slg->Nodes)[(*slotIn)++]=zero; */
/* 	         set_TmpMark1(oper1); */
/*                  return zero; */
/*             } */
/*           else {  */


/* 	    if (varPow_e(oper1) == 1) { */
/* 	      //new_sfixn_ini(one, 1); */
/*                   id = *slotIn; */
/*                   if(! id_of(one)) id_set(one, id); */
/* 	          IDsAry[id_of(oper1)]=id; */
/*                   (slg->Nodes)[(*slotIn)++]=one; */
/* 	          set_TmpMark1(oper1); */
/*                   return one; */
/*                } */
/* 	    else {  new_sfixn_ini(newoper1, varPow_e(oper1)); */
/*                     id = *slotIn; */
/*                     id_set(newoper1, id); */
/*                     (slg->Nodes)[(*slotIn)++]=newoper1; */
/*                     e = (varPow_e(oper1))-1; */
/*                     no = varPow_no(oper1); */
/*                     new_varpow_ini(newoper2, no , e); */
/*                     id = *slotIn; */
/*                     id_set(newoper2, id); */
/*                     (slg->Nodes)[(*slotIn)++]=newoper2; */
/*                     new_biProd_ini(newoper, newoper1, newoper2); */
/*                     id = *slotIn; */
/*                     id_set(newoper, id); */
/* 	            IDsAry[id_of(oper1)]=id; */
/*                     (slg->Nodes)[(*slotIn)++]=newoper; */
/* 	            set_TmpMark1(oper1); */
/*                     return newoper; */
/*             } */
/* 	  } */

	case t_biPlus:

	  //printf("t_biPlus\n");


	   new_biPlus_ini(newoper, 
			  getDerivOfAndInsertFromRoot_Hashing(slg, slotIn, 
							      biPlus_oper1(oper1),oper2, IDsAry,IDsAry2, hash_table), 
			  getDerivOfAndInsertFromRoot_Hashing(slg, slotIn, 
                          biPlus_oper2(oper1),oper2, IDsAry,IDsAry2, hash_table));
	   // if there is a equal node in the hash_table, then use the 
           // existing one.

	   tmpoper = searchNodeinHashTable(newoper, hash_table);
           if(tmpoper) {freeOperand(newoper); newoper = tmpoper; }

	             id = *slotIn;
             id_set(newoper, id);
	     IDsAry[id_of(oper1)]=id;

	     //printf("-7  --id = %ld\n", id);


             (slg->Nodes)[(*slotIn)++]=newoper;
	     set_TmpMark1(oper1);
             // inserting into hash_table ( to the beginning of a list)
             if((hash_table->table)[hashFunction(newoper)])
                  list_insert_after((hash_table->table)[hashFunction(newoper)], newoper);
             else (hash_table->table)[hashFunction(newoper)]=list_create(newoper);

             return newoper;
	case t_biSub:


	  //printf("t_biSub\n");

	     new_biSub_ini(newoper, 
			   getDerivOfAndInsertFromRoot_Hashing(slg, slotIn, 
                           biSub_oper1(oper1), oper2,IDsAry,IDsAry2, hash_table), 
			   getDerivOfAndInsertFromRoot_Hashing(slg, 
                                    slotIn, biSub_oper2(oper1), 
                                 oper2,IDsAry,IDsAry2, hash_table));


	     tmpoper = searchNodeinHashTable(newoper, hash_table);
             if(tmpoper) {freeOperand(newoper); newoper = tmpoper; }

             id = *slotIn;
             id_set(newoper, id);

	     //printf("-8  --id = %ld\n", id);

	     IDsAry[id_of(oper1)]=id;
             (slg->Nodes)[(*slotIn)++]=newoper;
	     set_TmpMark1(oper1);
             if((hash_table->table)[hashFunction(newoper)]) list_insert_after((hash_table->table)[hashFunction(newoper)], newoper);
             else (hash_table->table)[hashFunction(newoper)]=list_create(newoper);
             return newoper;
	case t_biProd:


	  //printf("t_biProd\n");


	  if(DEBUG22){
	    // printf("\n\n\n >>>> biprod operations: ");
	    // printOperand(oper1);
	  }
               new_biProd_ini(newoper1, 
                              getDerivOfAndInsertFromRoot_Hashing(slg, slotIn, 
					 biProd_oper1(oper1), oper2, IDsAry,IDsAry2, hash_table),
			      copy_operand_insert(slg, slotIn, biPlus_oper2(oper1),IDsAry2)
                              );

	     tmpoper = searchNodeinHashTable(newoper1, hash_table);
             if(tmpoper) {freeOperand(newoper1); newoper1 = tmpoper; }


                    id = *slotIn;
                    id_set(newoper1, id);
	            (slg->Nodes)[(*slotIn)++]=newoper1;

		    if((hash_table->table)[hashFunction(newoper1)])  list_insert_after((hash_table->table)[hashFunction(newoper1)], newoper1);
                    else  (hash_table->table)[hashFunction(newoper1)]=list_create(newoper1);


               new_biProd_ini(newoper2, 
                          copy_operand_insert(slg, slotIn, biPlus_oper1(oper1),IDsAry2),
			      getDerivOfAndInsertFromRoot_Hashing(slg, slotIn, 
					biProd_oper2(oper1), oper2, IDsAry,IDsAry2, hash_table)
			      );

	   // if there is a equal node in the hash_table, then use the 
           // existing one.
	     tmpoper = searchNodeinHashTable(newoper2, hash_table);
             if(tmpoper) {freeOperand(newoper2); newoper2 = tmpoper; }

                  id = *slotIn;
                  id_set(newoper2, id);
	          (slg->Nodes)[(*slotIn)++]=newoper2;
                  if((hash_table->table)[hashFunction(newoper2)]) list_insert_after((hash_table->table)[hashFunction(newoper2)], newoper2);
                  else (hash_table->table)[hashFunction(newoper2)]=list_create(newoper2);


	       new_biPlus_ini(newoper, newoper1, newoper2);
	   // if there is a equal node in the hash_table, then use the 
           // existing one.


	   //printf("* before searchNodeinHashTable()\n");
           //fflush(stdout);

	     tmpoper = searchNodeinHashTable(newoper, hash_table);
             if(tmpoper) {freeOperand(newoper); newoper = tmpoper; }


               id = *slotIn;
               id_set(newoper, id);
	       IDsAry[id_of(oper1)]=id;
	       (slg->Nodes)[(*slotIn)++]=newoper;
	       set_TmpMark1(oper1);
               //fprintf(stdout,"5 <<\n");
	       if ((hash_table->table)[hashFunction(newoper)]) list_insert_after((hash_table->table)[hashFunction(newoper)], newoper);
               else (hash_table->table)[hashFunction(newoper)]=list_create(newoper);
	       return newoper;	       
	default:
	        printf("default\n");
	        fprintf(stdout,"oper1 is an unknow type in function getDerivOfAndInsertFromRoot_Hashing()!\n");
                Throw 0;
    }

}



















/* SLG* */
/* createOneRowOfJMatrix_For_Lifting(operand *roots, */
/*                           int i, POLYMATRIX *mat,  */
/*                           POLYVECTOR_SLG *polyVec_SLG,  */
/*                           operand *vars, TriSet *ts, TriRevInvSet * tris,  */
/*                           sfixn N, MONTP_OPT2_AS_GENE * pPtr) */
/* { */
/*     int j, In=0, *slotIn=&In; */
/*     SLG *slg=ENTRYI_V(polyVec_SLG ,i); */
/*     SLG *derivSlg; */
/*     operand **tmpAry; */
/*     //=(operand **)my_malloc((slg->GN)*sizeof(operand *)); */
/*     operand one, zero; */
/*     new_sfixn_ini(zero, 0); */
/*     new_sfixn_ini(one, 1); */
/*   /\*   for(j=0; j<slg->GN; j++) { *\/ */

/* /\*          switch (type_of(NodeI(slg, j))) { *\/ */
/* /\* 	   case t_poly: *\/ */
/* /\* 	         fprintf(stdout,"can't be a t_poly now!.\n"); *\/ */
/* /\*                  Throw 32; *\/ */
/* /\*                  break; *\/ */
/* /\*            case t_sfixn: *\/ */
/* /\* 	   case t_var: *\/ */
/* /\* 	         tmpAry[j]=NULL; *\/ */
/* /\*                  break; *\/ */
/* /\*   	   case t_varpow: *\/ */
/* /\* 	         fprintf(stdout,"haven't implmented t_varpow type, now!.\n"); *\/ */
/* /\*                  Throw 32;	          *\/ */
/* /\*                  break; *\/ */
/* /\* 	   case t_biPlus: *\/ */
/* /\*          	 tmpAry[j]=(operand *)my_malloc((mat->N)*sizeof(operand));        *\/ */
/* /\*                  break; *\/ */
/* /\* 	   case t_biSub: *\/ */
/* /\* 	         tmpAry[j]=(operand *)my_malloc((mat->N)*sizeof(operand));  *\/ */
/* /\*                  break; *\/ */
/* /\* 	   case t_biProd: *\/ */
/* /\* 	         tmpAry[j]=(operand *)my_malloc((mat->N)*sizeof(operand));  *\/ */
/* /\*                  break; *\/ */
/* /\* 	   default: *\/ */
/* /\* 	        fprintf(stdout,"Binary operation won't be copied.\n"); *\/ */
/* /\*                 Throw 111; *\/ */
/* /\* 		return(derivSlg); *\/ */
/* /\*       } *\/ */

/* /\*     } *\/ */


/*     derivSlg = newSLG(((slg->GN)*N)<<4); */

/*     unsigned short int *IDsAry2= */
/*       (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int)); */

/*     clearTmpMarks(slg,2); */

/*     for(j=0; j<mat->N; j++){ */

/*       unsigned short int *IDsAry= */
/*       (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int)); */
/*       // to record the old-slg to new-slg, ID 2 ID relation for copy. */


/*       // In=0; */
      
/*       //derivSlg = newSLG((slg->GN)<<4); */
/*       clearTmpMarks(slg,1); */
      
/*       getDerivOfAndInsertFromRoot(zero, one, tmpAry, derivSlg, slotIn, ROOT_G(slg),  */
/*                              vars[(mat->N)-1-j], IDsAry, IDsAry2); */
/*       my_free(IDsAry); */

/*       if(j==(mat->N)-1) { shrinkG(derivSlg, In); setGN_G(derivSlg, In);} */

/*       //tmpslgAry[j] = derivSlg; */

/*       // setGN_G(tmpslgAry[j], In); */

/*       roots[j] = NodeI(derivSlg, In-1); */
/*       //roots[j]=ROOT_G(tmpslgAry[j]); */
/*       //getDerivOfG(ENTRYI_V(polyVec_SLG ,i), vars[(mat->N)-1-j]); */
/*     } */

/*     my_free(IDsAry2); */
    
/*     // for(j=0; j<slg->GN; j++) if(tmpAry[j]) my_free(tmpAry[j]); */
/*     //my_free(tmpAry); */

/*     return derivSlg; */
/* } */






/* //===============================================================> */
/* // End. */
/* //===============================================================> */





















//===================================================>









// obsolate
SLG*
createWholeJMatrix_For_Lifting(operand *roots,
                          int i, POLYMATRIX *mat, 
                          POLYVECTOR_SLG *polyVec_SLG, 
                          operand *vars, TriSet *ts, TriRevInvSet * tris, 
                          sfixn N, MONTP_OPT2_AS_GENE * pPtr)
{
  int iii, j, In=0, *slotIn=&In, size=0;
    SLG *slg;
unsigned short int *IDsAry2;

    SLG *derivSlg;
    //=(operand **)my_malloc((slg->GN)*sizeof(operand *));

// Change-Code: lift-0
//    operand one, zero;
    operand one=(operand)my_calloc(1, sizeof(operandObj));
    operand zero=(operand)my_calloc(1, sizeof(operandObj));

    new_sfixn_ini(zero, 0);
    new_sfixn_ini(one, 1);

    for(iii=0; iii<mat->M; iii++) size+=(ENTRYI_V(polyVec_SLG ,iii))->GN;

    derivSlg = newSLG((size*N)<<4);

for(iii=0; iii<mat->M; iii++) {


    slg= ENTRYI_V(polyVec_SLG ,iii);

    IDsAry2=
      (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int));

    clearTmpMarks(slg,2);

    for(j=0; j<mat->N; j++){

      unsigned short int *IDsAry=
      (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int));
      // to record the old-slg to new-slg, ID 2 ID relation for copy.


      // In=0;
      
      //derivSlg = newSLG((slg->GN)<<4);
      clearTmpMarks(slg,1);
      
      getDerivOfAndInsertFromRoot(zero, one, derivSlg, slotIn, ROOT_G(slg), 
                             vars[(mat->N)-1-j], IDsAry, IDsAry2);
      my_free(IDsAry);

      if((j==(mat->N)-1)&&(iii==((mat->M)-1))) { shrinkG(derivSlg, In); setGN_G(derivSlg, In);}

      //tmpslgAry[j] = derivSlg;

      // setGN_G(tmpslgAry[j], In);

      roots[iii*(mat->N)+j] = NodeI(derivSlg, In-1);
      //roots[j]=ROOT_G(tmpslgAry[j]);
      //getDerivOfG(ENTRYI_V(polyVec_SLG ,i), vars[(mat->N)-1-j]);
    }

    my_free(IDsAry2);
    

 }


    // for(j=0; j<slg->GN; j++) if(tmpAry[j]) my_free(tmpAry[j]);
    //my_free(tmpAry);

    return derivSlg;
}






//===============================================================>
// End.
//===============================================================>











// generate an Jacobean matrix for lifting with the help of a hash table.
SLG*
createWholeJMatrix_For_Lifting_Hashing(operand *roots,
                          int i, POLYMATRIX *mat, 
                          POLYVECTOR_SLG *polyVec_SLG, 
                          operand *vars, TriSet *ts, TriRevInvSet * tris, 
                          sfixn N, MONTP_OPT2_AS_GENE * pPtr)
{
  int iii, j, In=0, *slotIn=&In, size=0;
    SLG *slg;

    HASHTABLE *hash_table;
unsigned short int *IDsAry, *IDsAry2;
    SLG *derivSlg;
 
    //=(operand **)my_malloc((slg->GN)*sizeof(operand *));
    //operand one, zero;
    //new_sfixn_ini(zero, 0);
    //new_sfixn_ini(one, 1);


    for(iii=0; iii<mat->M; iii++) size+=(ENTRYI_V(polyVec_SLG ,iii))->GN;

    derivSlg = newSLG((size*N)<<4);

for(iii=0; iii<mat->M; iii++) {


    slg= ENTRYI_V(polyVec_SLG ,iii);

     IDsAry2=
      (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int));

    clearTmpMarks(slg,2);

    for(j=0; j<mat->N; j++){

      IDsAry=
      (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int));
      // to record the old-slg to new-slg, ID 2 ID relation for copy.


      // In=0;
      
      //derivSlg = newSLG((slg->GN)<<4);
      clearTmpMarks(slg,1);
      
      //printf("creating hash table!\n");
      //fflush(stdout);
      hash_table = newHashTable(HashModuli);
  
      //printf("before Derivative\n");
      //printSLG_Line(slg);



      getDerivOfAndInsertFromRoot_Hashing(derivSlg, slotIn, ROOT_G(slg),  vars[(mat->N)-1-j], IDsAry, IDsAry2, hash_table);


      //printf("after Derivative\n");
      //printSLG_Line_to(In, derivSlg);


      my_free(IDsAry);


      freeHashTable(hash_table);

      //printf("creating hash table!\n");
      //fflush(stdout);

      if((j==(mat->N)-1)&&(iii==((mat->M)-1))) { shrinkG(derivSlg, In); setGN_G(derivSlg, In);

	//printf("after Derivative\n");
	//printSLG_Line(derivSlg);

}


     


      //tmpslgAry[j] = derivSlg;

      // setGN_G(tmpslgAry[j], In);

      roots[iii*(mat->N)+j] = NodeI(derivSlg, In-1);
      //roots[j]=ROOT_G(tmpslgAry[j]);
      //getDerivOfG(ENTRYI_V(polyVec_SLG ,i), vars[(mat->N)-1-j]);
    }

    my_free(IDsAry2);
    

 }


    // for(j=0; j<slg->GN; j++) if(tmpAry[j]) my_free(tmpAry[j]);
    //my_free(tmpAry);

    return derivSlg;
}






//===============================================================>
// End.
//===============================================================>



















// N is number of variables.
// n is the moduli.
/**
 * operand2PolyOperand:
 * @slg: a DAG.
 * @oper: the source node the DAG.
 * @ts: a triangular set.
 * @tris: the modular-inverse of 'ts'.
 * @N: number of variables.
 * @pPtr: the information of the prime.
 * 
 * Convert the input DAG with nodes of different types, into nodes with 
 * only "C-CUBE" polynomial type.
 *
 * Return value: 
 **/
operand
operand2PolyOperand(SLG *slg, operand oper, TriSet *ts, TriRevInvSet * tris, sfixn N, MONTP_OPT2_AS_GENE * pPtr){
  preFFTRep *poly, *poly2;
        //operand newoper;
        int id;
        sfixn  tmp;

  	switch (type_of(oper)) {
	case t_poly:
	  //printSLG_Line(slg);
		//printf(" return oper, id is %d, type is %c.\n", id_of(oper), type_of(oper));
	  //printf(" -- 0 -- \n");
	  //fflush(stdout);
                
	        return(oper) ;
	case t_sfixn:
	  //printf(" -- 1 -- \n");
	  //fflush(stdout);
	  //t=gettime();
                 poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
                 InitOnePoly(poly, N, ts->bounds);
                 DATI(poly, 0) =  sfixn_val(oper);
                 id=id_of(oper);
                 //freeOperand(oper);
                 //new_poly_ini(newoper, poly);
                 (slg->Nodes)[id] = (operand)((POLYO *)oper);      
                 id_set((slg->Nodes)[id], id);
                 type_set((slg->Nodes)[id], t_poly);
	         poly_poly((slg->Nodes)[id])=poly;
                 //(slg->Nodes)[id] = newoper;
		 //printf(" -- 1 ---- \n");
		 //fflush(stdout);
                 //printf("di=%d.\n", id);
                 //fflush(stdout);
		 //time_sfixn+=gettime()-t;

		 //printf("END  -- 1 -- \n");
		 //fflush(stdout);


                 return (slg->Nodes)[id];
	case t_var:
	  //printf(" -- 2 -- \n");
	  //fflush(stdout);

	  //t=gettime();
                 poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
                 InitOnePoly(poly, N, ts->bounds);
		 
                 // reduced already or not.
                 if(ts->bounds[var_no(oper)]){
                   // don't need to reduce.
		   //printf(" -- 2.1 -- \n");
		   //fflush(stdout);
                   DATI(poly, CUMI(poly, var_no(oper))) = 1;
                 }
                 else{
		   //printf(" -- 2.2 -- \n");
		   //fflush(stdout);
                   // Evaluate this var by corresponding modulus.
		   //printf("N=%ld\n", N);
                   // printf("var_no(oper)=%ld\n", var_no(oper));
                   // fflush(stdout);
		   
                      fromtofftRep(var_no(oper), poly->accum, poly->data, 
                                CUM(ELEMI(ts, var_no(oper))),  
                                BUSZS(poly), DAT(ELEMI(ts, var_no(oper))));

                      negatePoly_1(poly, pPtr->P);
                 }

		 //printf(" -- 2.3 -- \n");
		 //fflush(stdout);

                 id=id_of(oper);
                 (slg->Nodes)[id] = (operand)((POLYO *)oper);      
                 id_set((slg->Nodes)[id], id);
                 type_set((slg->Nodes)[id], t_poly);
	         poly_poly((slg->Nodes)[id])=poly;


		 //printf(" -- 2,4 -- \n");
		 //fflush(stdout);

		 //time_var+=gettime()-t;

		 //printf("END -- 2 -- \n");
		 //fflush(stdout);

                 return (slg->Nodes)[id];               
	case t_varpow:
	  //printf("varpow case hasn't been test.\n delete the 'Throw 33' to test.\n");
	  // fflush(stdout);
       
	  // printf("<===============  %d^%d  ==================>\n", varPow_no(oper), varPow_e(oper));

                poly2 = (preFFTRep *)my_malloc(sizeof(preFFTRep));


                if((varPow_e(oper))>=(BDSI(ts, varPow_no(oper))+1)){
                    poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
                    tmp=BDSI(ts, varPow_no(oper));
                    BDSI(ts, varPow_no(oper))=varPow_e(oper);
	    	    InitOnePoly(poly, N, BDS(ts));
                    DATI(poly, CUMI(poly, varPow_no(oper))*(varPow_e(oper))) = 1;
	 	    BDSI(ts, varPow_no(oper))=tmp;
		    InitOnePoly(poly2, N, BDS(ts));

		    //printf("before reducing the poly\n");
		    //printPoly(poly);

                    MultiMod_ForLifting(N, poly2, poly, ts, tris, pPtr, SIZ(ELEMI(ts, N)));
                    EX_FreeOnePoly(poly);
		}
		else{
                    InitOnePoly(poly2, N, BDS(ts));
                    DATI(poly2, CUMI(poly2, varPow_no(oper))*(varPow_e(oper))) = 1;

		    //printf("before reducing the poly\n");
		    //printPoly(poly2);

		}



                id=id_of(oper);
                (slg->Nodes)[id] = (operand)((POLYO *)oper);      
                id_set((slg->Nodes)[id], id);
                type_set((slg->Nodes)[id], t_poly);
	        poly_poly((slg->Nodes)[id])=poly2;
                return (slg->Nodes)[id]; 

	case t_biPlus:

	  //printf(" -- 3-- \n");
	  //fflush(stdout);

                 poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
                 InitOnePoly(poly, N, ts->bounds);

		 //printf("3.11\n");
                 //fflush(stdout);

                 biPlus_oper1(oper)=
                    operand2PolyOperand(slg, biPlus_oper1(oper), ts, tris, N, pPtr);

                 biPlus_oper2(oper)=
                    operand2PolyOperand(slg, biPlus_oper2(oper), ts, tris, N, pPtr);

		 //t=gettime();

		 addPoly(N, poly,  poly_poly(biPlus_oper1(oper)), 
			 poly_poly(biPlus_oper2(oper)), pPtr->P);

		 //time_biPlus+=gettime()-t;


                 //printf("4.1\n");
                 id=id_of(oper);
                 //freeOperand(oper);
                 //new_poly_ini(newoper, poly);
                 //id_set(newoper, id);
                 //(slg->Nodes)[id] = newoper;
                 (slg->Nodes)[id] = (operand)((POLYO *)oper);      
                 id_set((slg->Nodes)[id], id);
                 type_set((slg->Nodes)[id], t_poly);
	         poly_poly((slg->Nodes)[id])=poly;

		 //printf(" END -- 3-- \n");
		 //fflush(stdout);

                 return (slg->Nodes)[id]; 
	case t_biSub:
	  //printf(" -- 5 -- \n");
	  //fflush(stdout);
		      // suppose children have been polified.



                 poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
                 InitOnePoly(poly, N, ts->bounds);
                 biSub_oper1(oper)=
                   operand2PolyOperand(slg, biSub_oper1(oper), ts, tris, N, pPtr);
                 biSub_oper2(oper)=
                   operand2PolyOperand(slg, biSub_oper2(oper), ts, tris, N, pPtr);                

		 //t=gettime();

                 //printf("5.0\n");
                 //fflush(stdout);
		 subPoly(N, poly,  poly_poly(biSub_oper1(oper)), 
			 poly_poly(biSub_oper2(oper)), pPtr->P);                 

		 //time_biSub+=gettime()-t;




                 //printf("5.1\n");
                 id=id_of(oper);
                 //freeOperand(oper);
                 //new_poly_ini(newoper, poly);
		 //id_set(newoper, id);
		//(slg->Nodes)[id] = newoper;
                 (slg->Nodes)[id] = (operand)((POLYO *)oper);      
                 id_set((slg->Nodes)[id], id);
                 type_set((slg->Nodes)[id], t_poly);
	         poly_poly((slg->Nodes)[id])=poly;

		 //printf(" End -- 5-- \n");
		 //fflush(stdout);

                 return (slg->Nodes)[id]; 
	case t_biProd:

	  //printf(" -- 6 -- \n");
	  //fflush(stdout);
		      // suppose children have been polified.

                 poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
                 InitOnePoly(poly, N, ts->bounds);
		      //DATI(poly, 0) =  sfixn_val(oper);

                 //printf("1->oper = ");
                 //printOperand(oper);


                 biProd_oper1(oper)=
                   operand2PolyOperand(slg, biProd_oper1(oper), ts, tris, N, pPtr);
                 biProd_oper2(oper)=
                   operand2PolyOperand(slg, biProd_oper2(oper), ts, tris, N, pPtr);  

                 //printf("2->oper = ");
                 //printOperand(oper);

		 //printf(" -- 6.0 -- \n");

            

	    //printOperand(biProd_oper1(oper));
	    //printOperand(biProd_oper2(oper));
            //fflush(stdout); 
            //printPoly(poly_poly(biProd_oper1(oper)));
            //printPoly(poly_poly(biProd_oper2(oper)));
		 //fflush(stdout);   
           
		 //t=gettime();

	  // EX_mul_Reduced(N, poly, poly_poly(biProd_oper1(oper)),
	  //                 poly_poly(biProd_oper2(oper)), ts, tris,pPtr);



          //printf("pp1=");
          //printPoly(poly_poly(biProd_oper1(oper)));
          //printf("pp2=");
          //printPoly(poly_poly(biProd_oper2(oper)));

        /*   printf("ts\n"); */
/*           printTriSet(ts); */

		 //printf("tris\n"); 
/*           printTriRevInvSet(tris); */

/*           printf("N=%ld\n", N); */
/*           printf("oper1=\n"); */
/*           printPoly(poly_poly(biProd_oper1(oper))); */
/*           printf("oper2=\n"); */
/*           printPoly(poly_poly(biProd_oper2(oper))); */


           //fflush(stdout); 

         

	       EX_mul_Reduced_ForLifting(N, poly, poly_poly(biProd_oper1(oper)),
	  				  poly_poly(biProd_oper2(oper)), ts, tris,pPtr);
          
	       //printf("finished\n");
               //fflush(stdout);


	       //time_biProd+=gettime()-t;

		 //printf(" -- 6.1 -- \n");
		 //fflush(stdout); 
                 id=id_of(oper);
                 //freeOperand(oper);
                 //new_poly_ini(newoper, poly);
                 //id_set(newoper, id);
                 //(slg->Nodes)[id] = newoper;
                 (slg->Nodes)[id] = (operand)((POLYO *)oper);      
                 id_set((slg->Nodes)[id], id);
                 type_set((slg->Nodes)[id], t_poly);
	         poly_poly((slg->Nodes)[id])=poly;

		 //printf(" End -- 6-- \n");
		 //fflush(stdout);

                 return (slg->Nodes)[id]; 
	default:
	  //printf("type = %c.\n", type_of(oper));
	        fprintf(stdout,"No this type!\n");
                Throw 1111;
		return oper;
	}



  

}



/**
 * evalVar2pt:
 * @slg: a DAG.
 * @varno: the index number of a variable "X_varno".
 * @val: the value to substitute the variable.
 * 
 * To evaluate the input DAG 'slg' at variable "X_varno" by value 'val'. 
 *
 * Return value: the image DAG after the evaluation.
 **/
SLG *
evalVar2pt(SLG *slg, int varno, sfixn val){
  SLG *newslg;
  unsigned short int *IDsAry2=
      (unsigned short int *)my_calloc(slg->GN, sizeof(unsigned short int));
  int i;

// Change-Code: lift-0
//    operand oper, newoper;
  operand oper=(operand)my_calloc(1, sizeof(operandObj));
  operand newoper=(operand)my_calloc(1, sizeof(operandObj));

  newslg=(SLG *)my_malloc(sizeof(SLG));
  newslg->GN=0;
  newslg->GE=0;
  newslg->Nodes=(operand *)my_calloc(slg->GN, sizeof(operand));
  clearTmpMarks(slg, 2);
  copy_operand_insert(newslg, &(newslg->GN),(ROOT_G(slg)), IDsAry2);

  assert( newslg->GN == slg->GN);

  for (i=0; i<newslg->GN; i++){
    oper=(newslg->Nodes)[i];
    if((type_of(oper)==t_var) && (var_no(oper)==varno)){
      //printOperand(oper);
          new_sfixn_ini(newoper, val);
          id_set(newoper, i);
          //printOperand(newoper);
          (newslg->Nodes)[i]=newoper;

          //printf("type_of(newoper) = %d\n", type_of(newoper));
    }
  }

  my_free(IDsAry2);
  return newslg;
}





/**
 * SLG2POLY:
 * @slg: a DAG.
 * @ts: a zero-dim triangular set.
 * @tris: the modular inverse of the trinagular set 'ts'.
 * @N: the number of variables.
 * @pPtr: the information the prime.
 * 
 * Convert the DAG polynomial modulo 'ts' into a C-Cube polynomial.
 *
 * Return value: a C-Cube polynomial.
 **/
preFFTRep *
SLG2POLY( SLG * slg, TriSet * ts, TriRevInvSet * tris, sfixn N, MONTP_OPT2_AS_GENE * pPtr){
  SLG * tmpSlg;
  preFFTRep *poly=(preFFTRep *)my_malloc(sizeof(preFFTRep));
  InitOnePoly(poly, N, BDS(ts));

  tmpSlg = removRedun(slg);

  ROOT_G(tmpSlg) = operand2PolyOperand(tmpSlg, ROOT_G(tmpSlg), ts, tris, N, pPtr);
 
  CopyOnePoly(poly, poly_poly(ROOT_G(tmpSlg)));

  freeSLG(tmpSlg);

  return poly;
}



/**
 * SLG2POLY:
 * @root: the root of the input DAG 'slg'.
 * @slg: a DAG.
 * @ts: a zero-dim triangular set.
 * @tris: the modular inverse of the trinagular set 'ts'.
 * @N: the number of variables.
 * @pPtr: the information the prime.
 * 
 * Convert the DAG polynomial modulo 'ts' into a C-Cube polynomial.
 *
 * Return value: a C-Cube polynomial.
 **/
preFFTRep *
SLG2POLY_ROOT( operand root, SLG * slg, TriSet * ts, TriRevInvSet * tris, sfixn N, MONTP_OPT2_AS_GENE * pPtr){
  
  preFFTRep *poly=(preFFTRep *)my_malloc(sizeof(preFFTRep));
  InitOnePoly(poly, N, ts->bounds);
  //tmpSlg = removRedun(slg);

  //printf("after Reducation:\n");
  //printSLG_Line(tmpSlg);
  //fflush(stdout);

  root = operand2PolyOperand(slg, root, ts, tris, N, pPtr);
  //printf("out of operand2PolyOperand.\n ");
  //fflush(stdout);
  CopyOnePoly(poly, poly_poly(root));

  //printf("root poly");
  //printPoly(poly);
  //fflush(stdout);


  // Throw 10000;
  // my_freeSLG(slg);
  //printf("leaving SLG2POLY");
  //fflush(stdout);
  return poly;
}




static 
void
degDirevOfPoly_inner(sfixn N, sfixn * res, sfixn * src, sfixn * dgs, sfixn * ccum, 
                     TriSet * ts, MONTP_OPT2_AS_GENE * pPtr, int varno){
  int i,j,d, offset;
  if(N==varno){
      d=shrinkDeg(dgs[N], src, ccum[N]);
      for(i=1; i<=d; i++){
        offset = (i-1)*ccum[N];
        for(j=0; j<ccum[N]; j++){
	   res[offset+j] = MulMod(src[offset+ccum[N]+j], i, pPtr->P); 
	}
      }
    return;
  }
  d=shrinkDeg(dgs[N], src, ccum[N]);
  for(i=0; i<=d; i++){
    degDirevOfPoly_inner(N-1, res+ccum[N]*i, src+ccum[N]*i, dgs, ccum, ts, pPtr, varno);
  }

}




preFFTRep *
degDirevOfPoly(preFFTRep *poly, int varno, TriSet * ts,  TriRevInvSet * tris, sfixn N, MONTP_OPT2_AS_GENE * pPtr){
  //printf("deriv===N=%d>  0.0\n",N);
  preFFTRep *newpoly1,*newpoly2;
if(varno<=N){
  newpoly1=(preFFTRep *)my_malloc(sizeof(preFFTRep));
  InitOnePoly(newpoly1, N, BUSZS(poly));
  degDirevOfPoly_inner(N, DAT(newpoly1), DAT(poly), BUSZS(poly), CUM(poly), 
                       ts, pPtr, varno);

  newpoly2=(preFFTRep *)my_malloc(sizeof(preFFTRep));
  InitOnePoly(newpoly2, N, ts->bounds);
  MultiMod(N, newpoly2, newpoly1, ts, tris, pPtr);
  freePoly(newpoly1);
  my_free(newpoly1);
 }
 else{
  newpoly2=(preFFTRep *)my_malloc(sizeof(preFFTRep));
  InitOnePoly(newpoly2, N, ts->bounds);
 }
//printf("deriv===>0.2\n");
  return newpoly2;
}




void
map_Y2one_1(POLYVECTOR_SLG * vec_slg, sfixn N, sfixn Y){
  int i, j;
  sfixn *map;
  SLG *slg;

// Change-Code: lift-0
//    operand oper;
  operand oper=(operand)my_calloc(1, sizeof(operandObj));

  map=(sfixn *)my_calloc(N+1, sizeof(sfixn));
  for(i=1; i<Y; i++){
    map[i]=i+1;
  }
  map[Y]=1;
  for(i=Y+1; i<=N; i++){
    map[i]=i;
  }

  for(i=0; i<M(vec_slg); i++){
     slg = ENTRYI_V(vec_slg, i);
     for(j=0; j<GN(slg); j++){
        oper = NodeI(slg, j);
        if(type_of(oper)==t_var ){
	  var_no(oper)=map[var_no(oper)];
	}
     }
  }
  
  my_free(map);

}





// 1 means yes, conconsistent.
/**
 * isInputSystemConsistent:
 * @N: number of variables.
 * @vec_slg: a vector DAG polynomials.
 * @ts: a zero-dim triangular set.
 * @pPtr: the information of the prime number 'p'.
 * @y0: the valuation value.
 * 
 * To check if the input system "vec_slg" is consistent with the input 
 * trinagular set "ts".
 *
 * Return value: 
 **/
int
isInputSystemConsistent(sfixn N, POLYVECTOR_SLG * vec_slg,   TriSet *ts, MONTP_OPT2_AS_GENE * pPtr, sfixn y0){
  TriRevInvSet *tris;
  POLYMATRIX *Fs;
  int i, j, consistent=1;
  sfixn *dgs;

   tris=(TriRevInvSet *)my_malloc(sizeof(TriRevInvSet));


  // --> RevInv06.


   dgs=(sfixn *)my_calloc(N+1, sizeof(sfixn));

   for(i=1; i<=N; i++){
     dgs[i]=BDSI(ts, i)+1;
     dgs[i]<<=1;
   }

   initTriRevInvSet( dgs, N, tris, ts);

   if(y0 != 0)  (DAT(ELEMI(ts, 1)))[0] = (pPtr->P) - y0;


   getRevInvTiSet(dgs, N, tris, ts, pPtr); 



   my_free(dgs);

   //printf("before SLG2PolyVecotr..\n");
   //fflush(stdout);

   //printf("N=%ld", N);

   //printSLG_Line(ENTRYI_V(vec_slg, 0));
   //printSLG_Line(ENTRYI_V(vec_slg, 1));
   //printf("SLG#3\n");
   //printSLG_Line(ENTRYI_V(vec_slg, 2));

  Fs=SLG2PolyVecotr(vec_slg, ts, tris, N, pPtr);

  //printf("after SLG2PolyVecotr..\n");
  //fflush(stdout);

           for(j=0; j<Fs->M; j++){
	     if (zeroPolyp( ENTRYI_M(Fs, j, 0)) !=1) {
                   consistent=0; 
                   break;}
           }

  if(y0 != 0)  (DAT(ELEMI(ts, 1)))[0] = 0;

  freeJMatrix(Fs);
  freeTriRevInvSet(tris);
  my_free(tris);
  return consistent;
}





// Never free the oldTriSet.
RFuncTriSet *
ReduceTriSetAndRFR(sfixn N, TriSet *oldTriSet, sfixn d, MONTP_OPT2_AS_GENE * pPtr){
      TriSet *Vts;
      TriRevInvSet *tris;
      RFuncTriSet *rfts;
      preFFTRep *T_;
      sfixn i;
      sfixn *dgs;

      dgs=(sfixn *)my_calloc(N+1, sizeof(sfixn));


      for(i=1; i<=N; i++){
	dgs[i]=BDSI(oldTriSet, i) + 1;
	dgs[i]<<=1;
      }

      Vts=EX_CopyOneTriSet(oldTriSet);
      for(i=2; i<=N-1; i++){
         T_ = direvativeMulti(ELEMI(Vts, i), pPtr);
         tris=(TriRevInvSet *)my_malloc(sizeof(TriRevInvSet));

     
  // --> RevInv07.
         initTriRevInvSet(dgs, i, tris, oldTriSet);
         getRevInvTiSet(dgs, i, tris, oldTriSet, pPtr); 


         MultiCoefPolyMul_1(i+1, T_, ELEMI(Vts, i+1), Vts, tris, pPtr);
         EX_FreeOnePoly(T_);
         NormalizeTriSetBDS(i, Vts);
         freeTriRevInvSet(tris);my_free(tris);
      }
      rfts=RFR_for_TriSet(Vts, d, pPtr);
      EX_freeTriSet(Vts);

      my_free(dgs);
      return rfts; 
}



// Y is the var has been specialized
// return rational funcation triangular set.
// input "ts" has been modidified (lifted).
//  lifting the least variable.
//  If TriSet = NULL, then Jacobi matrix inversion failed.

/**
 * UniNewtonLift:
 * @iterAddr: a pointer into a sfixn number to fetch the number of interation 
 * has been conducted in the lifting process.
 * @Y: is the var has been specialized .
 * @vec_slg: the input polynomial system (a vector DAGs).
 * @ts: a triangular set.
 * @N: the number of variables.
 * @pPtr: the information of the prime number. 
 *
 * Return value: the lifted trinagular set with rational function coefficients
 **/
RFuncTriSet *UniNewtonLift(int *iterAddr, sfixn Y, POLYVECTOR_SLG * vec_slg, TriSet * ts, 
                   sfixn N, MONTP_OPT2_AS_GENE * pPtr){
  POLYMATRIX *I0 = NULL, *J0, *f_1_n, *g_1_n, *K, *h_1_n, *tmp1M, *tmp2M;
  TriRevInvSet * tris;
  int i=0,finish=0, k;
  int d;
  TriSet *newts=NULL;
  //TriRevInvSet * newtris;
  sfixn pt=0;
  RFuncTriSet *rfts;
  // POLYMATRIX *Fs;
  //  double t_TMP;
  sfixn *dgs;

  signal(SIGINT,catch_intr);

  dgs=(sfixn *)my_calloc(N+1, sizeof(sfixn));


  for(k=1; k<=N; k++){
	dgs[k]=BDSI(ts, k) + 1;

  }


  tris=(TriRevInvSet *)my_malloc(sizeof(TriRevInvSet));


  while(!finish){
    //printf("iteration#%d.\n", i+1);
    // if(i==8) {printf("Lifting Aborted to check errors!\n"); Throw 1011;};

    *iterAddr = i +1;

    // printf("iterAddr=%d\n", *iterAddr);
    if(Interrupted==1) {
	     freeJMatrix(I0);
             my_free(dgs);
             //Interrupted=0;
             return NULL;
    }



    // if(i==5) Throw 0;

      if(i == 0){
 
	  // printf("dead-1\n");

	  if(DEBUG20){
            printf("<<<<--00----ts:>>>>\n");
            printTriSet(ts);
            fflush(stdout);}

          //Throw 10000;

  // --> RevInv08.


          initTriRevInvSet(dgs, N, tris, ts);
          getRevInvTiSet(dgs, N, tris, ts, pPtr);



	  if(DEBUG20){
            printf("<<<<--01----tris:>>>>\n");
	    printTriRevInvSet(tris);
            fflush(stdout);
	  }



          J0 = createJMatrix_ForLifting(vec_slg, ts, tris, N, pPtr);



	  if(DEBUG20){
            printf("<<<<--02----J0>>>>\n");
	    printJMatrix(J0);
            fflush(stdout);
	  }

          I0 = INVJMatrix(N, J0, ts, tris, pPtr);


          if(I0 == NULL){
            //printf("----1.1----\n");
            fflush(stdout);
            freeTriRevInvSet(tris);my_free(tris);
	    freeJMatrix(J0);
            //printf("----1.2----\n");
            //fflush(stdout);
            return NULL;

	  }
          


	  if(DEBUG20){
            printf("<<<<--03----I0>>>>\n");
	    printJMatrix(I0);
            fflush(stdout);
	  }


          //Throw 10000;

          freeTriRevInvSet(tris);my_free(tris);
          freeJMatrix(J0);
 
          
      }
    else{
          
           //==================================
           tris=(TriRevInvSet *)my_malloc(sizeof(TriRevInvSet));

           dgs[1]=2*dgs[1]+1;
  // --> RevInv09.
           initTriRevInvSet( dgs, N, tris, ts);
           getRevInvTiSet(dgs, N, tris, ts, pPtr); 
 
	   //=================================

	   //t_TMP=gettime();
          J0 = createJMatrix_ForLifting(vec_slg, ts, tris, N, pPtr);
          //t01+=gettime()-t_TMP;



          if(Interrupted==1) {

	     freeTriRevInvSet(tris);my_free(tris);
             freeJMatrix(J0);
             my_free(dgs);
             //Interrupted=0;
             return NULL;
           }



	  if(DEBUG20){
            printf("<<<<--03----J0>>>>\n");
	    printJMatrix(J0);
            fflush(stdout);
	  }

	  //t_TMP=gettime();
          tmp1M=mulJMatrix(N, I0, J0, ts, tris, pPtr);
	  //t02+=gettime()-t_TMP;

	  if(DEBUG20){
             printf("<<<<--03.1.1---- tmp1M>>>>\n");
	     printJMatrix(tmp1M);
             fflush(stdout);
	   }

          freeJMatrix(J0);        

	  //t_TMP=gettime();
          tmp2M=mulJMatrix(N, tmp1M, I0, ts, tris, pPtr);
	  //t02+=gettime()-t_TMP;

          freeJMatrix(tmp1M);


	  scalarMulJMatrix_1(2, I0, pPtr);

	  //t_TMP=gettime();
          subJMatrix_1(N, I0, tmp2M, pPtr);
	  //t03+=gettime() - t_TMP;


	   if(DEBUG20){
             printf("<<<<--03.5---- I0>>>>\n");
	     printJMatrix(I0);
             fflush(stdout);
	   }


          freeJMatrix(tmp2M);                    

          freeTriRevInvSet(tris);my_free(tris);
     }


      //t_TMP=gettime();          
           LiftTinTriSet(N, ts, i, pPtr); 
	   //t04+=gettime(0)-t_TMP;

	   // printf("dead-1.\n");
	   //fflush(stdout);

	   if(DEBUG20){
             printf("<<<<--04----lifted ts>>>>\n");
	     printTriSet(ts);
             fflush(stdout);
	   }


           I0=increaseMatrix_ForLifting(N, I0, ts);


	   if(DEBUG20){
             printf("<<<<--05----increased I0>>>>\n");
	     printJMatrix(I0);
             fflush(stdout);
	   }


           //==================================
           tris=(TriRevInvSet *)my_malloc(sizeof(TriRevInvSet));
 
	   //t_TMP=gettime();

  // --> RevI<<nv10. 
	   initTriRevInvSet(dgs, N, tris, ts);
	   getRevInvTiSet(dgs, N, tris, ts, pPtr); 
	   //t05+=gettime()-t_TMP;


 	   if(DEBUG20){
             printf("<<<<--06---- new tris:>>>>\n");
	     printTriRevInvSet(tris);
             fflush(stdout);
	   }

	   //=================================

	   //t_TMP=gettime();
           f_1_n=SLG2PolyVecotr(vec_slg, ts, tris, N, pPtr);
	   //t06+=gettime()-t_TMP;

 	   if(DEBUG20){
             printf("<<<<--07---- f_1_n>>>>\n");
	     printJMatrix(f_1_n);
             fflush(stdout);
	   }

	   //t_TMP=gettime();
           g_1_n=mulJMatrix(N, I0, f_1_n, ts, tris, pPtr);
	   //t02+=gettime()-t_TMP;


 	   if(DEBUG20){
             printf("<<<<--08---- g_1_n>>>>\n");
	     printJMatrix(g_1_n);
             fflush(stdout);
	   }

           freeJMatrix(f_1_n);

	   //t_TMP=gettime();
	   K=createJMatrix_PolyForlifting(ts, tris, N, pPtr);
	   //t01+=gettime()-t_TMP;

 	   if(DEBUG20){
             printf("<<<<--09---- K>>>>\n");
	     printJMatrix(K);
             fflush(stdout);
	   }

	   //t_TMP=gettime();
           h_1_n=mulJMatrix(N, K, g_1_n, ts, tris, pPtr);
	   //t02+=gettime()-t_TMP;


 	   if(DEBUG20){
             printf("<<<<--10---- h_1_n>>>>\n");
	     printJMatrix(h_1_n);
             fflush(stdout);
	   }

           freeJMatrix(g_1_n);
           freeJMatrix(K); 

	   //t_TMP=gettime();   
           AddHs2TriSet(N, ts, h_1_n, pPtr);
	   //t07+=gettime(0)-t_TMP;


 	   if(DEBUG20){
             printf("<<<<--11---- updated ts>>>>\n");
	     printTriSet(ts);
             fflush(stdout);
	   }

           freeJMatrix(h_1_n); 
 	   freeTriRevInvSet(tris);my_free(tris);
         
           i++;

           d=1<<i;
      
           // the branch is improved one which makes the lifting stop earlier.
	   rfts=ReduceTriSetAndRFR(N, ts, d, pPtr);
           if(rfts==NULL) {
	     //printf("rational function reconstruction failed.\n"); 
             //fflush(stdout); 
                 continue;}

           while (!pt) pt=rand()%pPtr->P;
           newts=EvalRFTriSetAtZeroForSmallestVarAndMakeThefirstOneBeYAndMonicize(rfts->numeris, rfts->denomis, pPtr, pt);
           if(newts == NULL){
	           EX_freeTriSet(newts);
                   freeRFT(rfts);
                   freeJMatrix(I0);
	   }
  
	   finish = isInputSystemConsistent(N, vec_slg,  newts, pPtr, 0);

	   EX_freeTriSet(newts);

           if (!finish) freeRFT(rfts);




  }
  
  freeJMatrix(I0);
  my_free(dgs);
  return rfts;

}












// Subroutine for shiftInputSystem().
// there require the ID of a node exactly corresponds to it's location in the array.
static 
void
incr_ref_by_m(SLG *slg, operand oper, int varno, int m){
// Change-Code: lift-0
//    operand node;
  operand node=(operand)my_calloc(1, sizeof(operandObj));
	switch (type_of(oper)) {
	case t_sfixn:
	case t_var:
	case t_varpow:
	  // no refenrece to be updated
                 return;
	case t_biPlus:
	  node = biProd_oper1(oper);
          if((type_of(node)==t_var) && (var_no(node)==varno)){
	    biPlus_oper1(oper)= NodeI(slg, id_of(node)+m);
          }
	  node = biPlus_oper2(oper);
          if((type_of(node)==t_var) && (var_no(node)==varno)){
	    biPlus_oper2(oper)= NodeI(slg, id_of(node)+m);
          }
          return;
	case t_biSub:
	  node = biSub_oper1(oper);
          if((type_of(node)==t_var) && (var_no(node)==varno)){ 
	    biSub_oper1(oper)= NodeI(slg, id_of(node)+m);
          }
	  node = biSub_oper2(oper);
          if((type_of(node)==t_var) && (var_no(node)==varno)){
	    biSub_oper2(oper)= NodeI(slg, id_of(node)+m);
          }
          return;
	case t_biProd:
	  node = biProd_oper1(oper);
          if((type_of(node)==t_var) && (var_no(node)==varno)){
	    biProd_oper1(oper)= NodeI(slg, id_of(node)+m);
          }
	  node = biProd_oper2(oper);
          if((type_of(node)==t_var) && (var_no(node)==varno)){
	    biProd_oper2(oper)= NodeI(slg, id_of(node)+m);
          }
          return;
	default:
	  fprintf(stdout,"type %d is not ok.\n",type_of(oper) );
                Throw 112;
		return;
	}
}







// subs(X_Y=X_Y+y0, vec_slg). 

/**
 * shiftInputSystem:
 * @vec_slg: a vector of DAGs.
 * @Y: the variable to be shifted.
 * @y0: the distance to be shifted.
 * @pPtr: the information for the prime number.
 * 
 * Substitute variable Y by Y+y0 for all DAG polynomials in the input vector.  
 *
 * Return value: a vector of DAGs of the shift.
 **/
POLYVECTOR_SLG *
shiftInputSystem(POLYVECTOR_SLG *vec_slg, sfixn Y, sfixn y0, MONTP_OPT2_AS_GENE *pPtr){
  POLYVECTOR_SLG *new_vec_slg;								      int i, j, incr=0, id;
  SLG *slg, *newslg;

// Change-Code: lift-0
//    operand node;
  operand node=(operand)my_calloc(1, sizeof(operandObj));
  // newID = increaseVec[oldID];
  int * increaseVec;
  new_vec_slg=(POLYVECTOR_SLG *)my_malloc(sizeof(POLYVECTOR_SLG));
  new_vec_slg->M = vec_slg->M;
  new_vec_slg->entries = (SLG **)my_malloc((new_vec_slg->M)*sizeof(SLG *));

   for(i=0; i<M(vec_slg); i++){
     incr=0;
     slg = ENTRYI_V(vec_slg, i);

     increaseVec=(int *)my_calloc(GN(slg),sizeof(int));

     // (0) create increase vector.
     for(j=0; j<GN(slg); j++){
       node = NodeI(slg, j); 
       increaseVec[j]=id_of(node) + 2*incr;
       if((type_of(node)==t_var) && var_no(node)==Y){
           incr++;
       }
     }

     // (1) make a copy
     newslg =  (SLG *)my_malloc(sizeof(SLG));
     // incr is the number of node to expand. (1 -> 3).
     GN(newslg)=0;
     GE(newslg)=0;
     newslg->Nodes=(operand *)my_calloc(GN(slg)+2*incr, sizeof(operand));
     // make a copy.
     removRedun_1(newslg, slg);
     GN(newslg)=GN(slg)+2*incr;
     GE(newslg)=GE(newslg)+2*incr;

     //removRedun(SLG * slg);



     // (2) sliding other nodes.
     for(j=GN(slg)-1; j>=0; j--){
       NodeI(newslg, increaseVec[j])=NodeI(newslg, j);
       id_set(NodeI(newslg, increaseVec[j]), increaseVec[j]);
     }


     // (3) inserting new nodes (also variable "y")
     for(j=0; j<GN(slg); j++){
       node = NodeI(slg, j); 

       if((type_of(node)==t_var) && var_no(node)==Y){
	 // the origin var.
  	    id = increaseVec[j];
	    // (newslg->Nodes)[id] = alloc_operand(t_var);
	    // create_var((newslg->Nodes)[id], 1, id);
 	 // the y0.
            id = id+1; 
            (newslg->Nodes)[id] = alloc_operand(t_sfixn);
	    create_sfixn((newslg->Nodes)[id], y0, id);
 	 // the plus of var and y0.
            id = id+1;
            (newslg->Nodes)[id] = alloc_operand(t_biPlus);
	    create_biPlus(newslg->Nodes, (newslg->Nodes)[id], id-2, id-1, id);
            incr++;
       }
     }

  
     //(4) just pointer from "to var" to "to var+y0).
     for(j=0; j<GN(newslg); j++){
         node = NodeI(newslg, j);    
        if((type_of(node)==t_var) && var_no(node)==Y){
	  j+=2;
	}
	else{
             incr_ref_by_m(newslg, NodeI(newslg, j), Y, 2);
	}
     }

     my_free(increaseVec);
     
     ENTRYI_V(new_vec_slg, i)=newslg;
   }

   return new_vec_slg;
}





// computes g = f(x+c)
//void shiftPolynomial(ZZ_pX& g, const ZZ_pX& f, const ZZ_p c){
// subroutine for shiftTriSet().
void shiftPolynomial(sfixn degg, sfixn *g, sfixn degf, sfixn *f, sfixn c, MONTP_OPT2_AS_GENE * pPtr){
  sfixn i;
  //  ZZ_pX tmp1, tmp2, tmp3;
  sfixn *tmp1, *tmp2, *tmp3; 
  //  ZZ_p powC = to_ZZ_p(1);
  sfixn powC=1;
  //  ZZ_p fact = to_ZZ_p(1);
  sfixn fact=1;
  //  long d = deg(f);
  sfixn d = degf;
  //  tmp1.rep.SetLength(d+1);
  tmp1=(sfixn *)my_calloc(d+1, sizeof(sfixn));
  //  tmp2.rep.SetLength(d+1);
  tmp2=(sfixn *)my_calloc(d+1, sizeof(sfixn));
  //  for (long i = 0; i <= d; i++){
  for ( i = 0; i<= d ; i++){
    //    div(tmp1.rep[i], powC, fact);
    tmp1[i] = DivMod(powC, fact, pPtr->P);
    //    mul(tmp2.rep[d-i], fact, coeff(f, i));
    tmp2[d-i] = MulMod(fact, f[i], pPtr->P);
    //    mul(powC, powC, c);
    powC = MulMod( powC,  c, pPtr->P);
    //mul(fact, fact, 1+i);
    fact = MulMod( fact, 1+i, pPtr->P);
    //    }
  }
  //  tmp1.normalize();
  //normalize_1(d, tmp1, pPtr);
  //  tmp2.normalize();
  // normalize_1(d, tmp2, pPtr);
  //  mul(tmp3, tmp1, tmp2);
  tmp3=(sfixn *)my_calloc(d+d+1, sizeof(sfixn));
  EX_Mont_FFTMul(d+d, tmp3, d, tmp1, d, tmp2, pPtr);
  //  fact = to_ZZ_p(1);
  fact = 1;
  //  g.rep.SetLength(d+1); ??
  //for (long i = 0; i <= d; i++){
  for (i=0; i <= d; i++){
    //    div(g.rep[i], coeff(tmp3, d-i), fact);
    g[i] = DivMod(tmp3[d-i], fact, pPtr->P);
    //    mul(fact, fact, 1+i);
    fact = MulMod(fact, 1+i, pPtr->P);
    //}
  }
  //  g.normalize(); // ?
  //normalize_1(degg, g, pPtr);
  //}
}








// return subs(x=x+c, ts)
// Never free a parameter object!!! from now on.
/**
 * shiftTriSet:
 * @oldts: a input triangular set.
 * @c: a sfixn number.
 * @pPtr: the information of prime p.
 * 
 * subsitute x by x+c for the triangular set.
 *
 * Return value: the new triangular set.
 **/
TriSet *
shiftTriSet(TriSet *oldts, sfixn c, MONTP_OPT2_AS_GENE * pPtr){
  sfixn N=N(oldts);
  int i, j;
  TriSet *newts;
  sfixn *dgs;
  preFFTRep *newPoly, *oldPoly;
  sfixn degg, *g, degf, *f;

  dgs = bounds2dgs(oldts);
  newts = Ex_InitDenseTriSet(N, dgs, pPtr);
  
  

  for(i=1; i<=N; i++){
    oldPoly = ELEMI(oldts, i);
    newPoly = ELEMI(newts, i);

    for( j=0; j<SIZ(oldPoly); j+=BUSZSI(oldPoly, 1)+1){
      degg=degf=BUSZSI(oldPoly, 1); 
      g=DAT(newPoly)+j;
      f=DAT(oldPoly)+j;
      degg=shrinkDegUni(degg, g);
      degf=shrinkDegUni(degf, f);
      shiftPolynomial(degg, g, degf, f, c, pPtr);
   }
  }
  my_free(dgs);
  return newts;
}





/**
 * RemoveDenominators:
 * @rfts: a trinagular set with univariate rational function coefficients.
 * @pPtr: the information of the prime number.
 *
 *  Removed all the denominators in all rational function coefficients.
 *
 * Return value: a trinagular with remove all the denominators in 
 *               the univariate rational function coefficients.
 **/
TriSet *
RemoveDenominators(RFuncTriSet * rfts,  MONTP_OPT2_AS_GENE *pPtr){
  int i, j, k;
  sfixn N=N(rfts->numeris);
  preFFTRep *numPoly=NULL, *denPoly=NULL, *newPoly=NULL;
  sfixn **Lcms, *Lcm, dgLcm, *dgLcms, *newLcm, dgnewLcm, *dgnewLcmAddr=&dgnewLcm, df, *f, dg, *g;
  sfixn newd1=0, dgProd, *prod;
  TriSet *newts;
  sfixn *dgs, dgQ, *Q;

  assert(N == N(rfts->denomis));

  Lcms=(sfixn **)my_calloc(N+1, sizeof(sfixn*));
  dgLcms=(sfixn *)my_calloc(N+1, sizeof(sfixn));

  for(i=1; i<=N; i++){
    dgLcm=0;
    Lcm=(sfixn *)my_calloc(1, sizeof(sfixn));
    Lcm[0]=1;
    denPoly = ELEMI(rfts->denomis, i);
    for( j=0; j<SIZ(denPoly); j+=BUSZSI(denPoly, 1)+1){
      df=BUSZSI(denPoly, 1);
      f=DAT(denPoly)+j;
      df=shrinkDegUni(df, f);
      if(isZeroVec(df+1, f)) continue;
      newLcm=LcmPolyPair(dgnewLcmAddr, dgLcm, Lcm, df, f, pPtr);
      my_free(Lcm);
      dgLcm=dgnewLcm;
      Lcm=newLcm;
    }
    Lcms[i]=Lcm;
    dgLcms[i]=dgLcm;
    if(newd1<dgLcms[i]) newd1=dgLcms[i];
  }


  newd1=newd1+ BUSZSI(ELEMI(rfts->denomis, 1), 1);

  newts=(TriSet *)my_malloc(sizeof(TriSet));
  dgs=(sfixn *)my_calloc(N, sizeof(sfixn));
  for(i=0; i<N; i++){
    dgs[i]=BDSI(rfts->numeris, i+1)+1;
  }
  dgs[0]=newd1;

  initDenseTriSet(N, dgs, newts, pPtr);

  for(i=1; i<=N; i++){
    numPoly = ELEMI(rfts->numeris, i);
    denPoly = ELEMI(rfts->denomis, i);
    newPoly = ELEMI(newts, i);
    for( j=0, k=0; j<SIZ(numPoly); j+=BUSZSI(numPoly, 1)+1, k+=BDSI(newts ,1)+1){
      df=BUSZSI(numPoly, 1);
      f=DAT(numPoly)+j;
      dg=BUSZSI(denPoly, 1);
      g=DAT(denPoly)+j;
      df=shrinkDegUni(df, f);
      dg=shrinkDegUni(dg, g);

      if(isZeroVec(df+1, f) || isZeroVec(dg+1, g)) continue;

      dgQ=dgLcms[i]-dg;
      Q= (sfixn *) my_calloc(dgQ+1, sizeof(sfixn));
      UniQuo(dgQ, Q, dgLcms[i], Lcms[i], dg, g, pPtr );
      dgQ=shrinkDegUni(dgQ, Q);  
      dgProd=df+dgQ;
      prod=DAT(newPoly)+k;
      EX_Mont_FFTMul(dgProd, prod, dgQ, Q, df, f, pPtr);
      my_free(Q);
   }
  }
 
  if(Lcms){
     for(i=1; i<=N; i++){  if(Lcms[i]) my_free(Lcms[i]);}
     my_free(Lcms);
  }
  my_free(dgLcms);
  my_free(dgs);

  return newts;

}




// returning NULL means failed.
// x^d is m in RFT.
// d/2 is d in RFT.
/**
 * RFR_for_TriSet:
 * @ts: triangular set.
 * @d: the degree bound to stop the rational function reconstruction.
 * @pPtr: the information for the prime number p.
 * 
 * Return value: 
 **/
RFuncTriSet *
RFR_for_TriSet(TriSet *ts, sfixn d, MONTP_OPT2_AS_GENE * pPtr)
{
  int i, j, k, check, stop;
  sfixn *dgs, N, sizOfDenOrNum;
  RFuncTriSet * RFuncTs=(RFuncTriSet *)my_malloc(sizeof(RFuncTriSet));
  TriSet *tnumPtr=(TriSet *)my_malloc(sizeof(TriSet)), *tdenPtr=(TriSet *)my_malloc(sizeof(TriSet));
  preFFTRep *poly, *denpoly, *numpoly;
  sfixn *fData, *vData, *gcdData, *modulus, dmodulus, df;
        sfixn dv, *dvAddr, dgcd, *dgcdAddr;
  dmodulus=d;
  modulus=(sfixn *)my_calloc(dmodulus+1,sizeof(sfixn));
  modulus[dmodulus]=1;
  RFuncTs->numeris= tnumPtr;
  RFuncTs->denomis= tdenPtr;
  N=N(ts);
  dgs=(sfixn *)my_calloc(N, sizeof(sfixn));
  for(i=0; i<N; i++){
    dgs[i]=BDSI(ts, i+1)+1;
  }

  //printf("dgs=");
  // printVec(N-1, dgs);

  // we suppose the num/den both have the bound.
  dgs[0]=dgs[0];

  initDenseTriSet(N, dgs, tnumPtr, pPtr);
  initDenseTriSet(N, dgs, tdenPtr, pPtr);
  
  sizOfDenOrNum=dgs[0];

  for(i=2; i<=N; i++){
    poly=ELEMI(ts, i);
    fData=DAT(poly);
    denpoly=ELEMI(tdenPtr, i);
    vData=DAT(denpoly);
    numpoly=ELEMI(tnumPtr, i);
    gcdData=DAT(numpoly);
    for(j=0, k=0; j<SIZ(poly); j+=BUSZSI(poly,1)+1, k+=sizOfDenOrNum){

      df=BUSZSI(poly, 1);
      df=shrinkDegUni(df, fData+j);
      dv=dgs[0]; dvAddr=&dv; dgcd=0; dgcdAddr=&dgcd;
      stop=d/2 ;      
      stop+=6;
      check=ExGcd_Uni_RFR(stop, vData+k, dvAddr, gcdData+k, dgcdAddr, modulus, dmodulus, fData+j, df, pPtr);
      if(check!=0){
      freeRFT(RFuncTs);
      my_free(dgs); my_free(modulus);
      return NULL;
      }
   }


  }
  
  my_free(dgs); my_free(modulus);
  return RFuncTs;
}




void
EvalRFPolyAtZeroForSmallestVar(preFFTRep *despoly, preFFTRep *srcnumpoly, preFFTRep *srcdenpoly, sfixn pt, sfixn p){
  int i, j;
  sfixn n,d;
  for(i=0, j=0; i<SIZ(srcnumpoly); i+=BUSZSI(srcnumpoly, 1)+1, j++){
 
    if(! isZeroVec(BUSZSI(srcnumpoly, 1)+1, (DAT(srcdenpoly))+i)){
      //assert((DAT(srcdenpoly))[i] != 0);
      n=SlowEvaluation1pt(BUSZSI(srcnumpoly, 1), DAT(srcnumpoly)+i, pt, p);
      d=SlowEvaluation1pt(BUSZSI(srcdenpoly, 1), DAT(srcdenpoly)+i, pt, p);    
      (DAT(despoly))[j] =MulMod(n, inverseMod(d, p), p);
    }
  }
}


TriSet *
EvalRFTriSetAtZeroForSmallestVarAndMakeThefirstOneBeY(TriSet *tsnum, TriSet *tsden, MONTP_OPT2_AS_GENE *pPtr, sfixn pt){
  int i;
  sfixn N=N(tsnum), *dgs;
  preFFTRep *despoly, *srcnumpoly, *srcdenpoly;
  TriSet * rests;
  rests=(TriSet *)my_malloc(sizeof(TriSet));
 
  // tsnum has the same bounds as tsden.

  dgs=(sfixn *)my_calloc(N, sizeof(sfixn));
  for(i=0; i<N; i++){
    dgs[i]=BDSI(tsnum, i+1)+1;
  }
  // going to make the first poly in rests to be Y.
  dgs[0]=1;
 
  initDenseTriSet( N, dgs, rests,  pPtr);


  for(i=2; i<=N; i++){
    despoly=ELEMI(rests, i);
    srcnumpoly=ELEMI(tsnum, i);
    srcdenpoly=ELEMI(tsden, i);
    EvalRFPolyAtZeroForSmallestVar(despoly, srcnumpoly, srcdenpoly, pt, pPtr->P);    
  }
  my_free(dgs);

  // put Y into the new ts.
  DAT(ELEMI(rests, 1))[1]=1;
  DAT(ELEMI(rests, 1))[0]=(pPtr->P)-pt;
  return rests;
}




TriSet *
EvalRFTriSetAtZeroForSmallestVarAndMakeThefirstOneBeYAndMonicize(TriSet *tsnum, TriSet *tsden, MONTP_OPT2_AS_GENE *pPtr, sfixn pt){
  int i, good;
  sfixn N=N(tsnum), *dgs;
  preFFTRep *despoly, *srcnumpoly, *srcdenpoly;
  TriSet * rests;
  rests=(TriSet *)my_malloc(sizeof(TriSet));
 
  // tsnum has the same bounds as tsden.

  dgs=(sfixn *)my_calloc(N, sizeof(sfixn));
  for(i=0; i<N; i++){
    dgs[i]=BDSI(tsnum, i+1)+1;
  }
  // going to make the first poly in rests to be Y.
  dgs[0]=1;
 
  initDenseTriSet( N, dgs, rests,  pPtr);

  for(i=2; i<=N; i++){
    despoly=ELEMI(rests, i);
    srcnumpoly=ELEMI(tsnum, i);
    srcdenpoly=ELEMI(tsden, i);
    EvalRFPolyAtZeroForSmallestVar(despoly, srcnumpoly, srcdenpoly, pt, pPtr->P);    
  }
  my_free(dgs);


  // put Y into the new ts.
  DAT(ELEMI(rests, 1))[1]=1;
  DAT(ELEMI(rests, 1))[0]=(pPtr->P)-pt;



	      good = MonicizeTriSet_1(N, rests, pPtr);
              if(good == -1){
                printf("Can't be monicized\n"); fflush(stdout);
                EX_freeTriSet(rests);
                return NULL;
	      }




  return rests;
}











void
freeRFT(RFuncTriSet *srft){

  if(srft !=NULL){

 if(srft->numeris != NULL){
   freeTriSet(srft->numeris); my_free(srft->numeris);
 }
 if(srft->denomis != NULL){
   freeTriSet(srft->denomis); my_free(srft->denomis);
 }
    my_free(srft);
  }
}



/**
 * EvalPolyAtZeroForSmallestVar:
 * @despoly: (output) the output C-Cube polynomial.
 * @srcpoly: the input C-Cube polynomial.
 * @pt: a evaluation point.
 * @p: a prime number.
 * 
 * To evaluate 'scrpoly' at smallest variable with value 'pt'.
 * 'despoly' is the output result after evluation.
 * Return value: 
 **/
void
EvalPolyAtZeroForSmallestVar(preFFTRep *despoly, preFFTRep *srcpoly, sfixn pt, sfixn p){
  int i, j;
  for(i=0, j=0; i<SIZ(srcpoly); i+=BUSZSI(srcpoly, 1)+1, j++){

    (DAT(despoly))[j] = SlowEvaluation1pt(BUSZSI(srcpoly, 1), DAT(srcpoly)+i, pt, p);

  }
 
}


TriSet *
EvalTriSetAtZeroForSmallestVarAndMakeThefirstOneBeY(TriSet *ts, MONTP_OPT2_AS_GENE *pPtr, sfixn pt){
  int i;
  sfixn N=N(ts), *dgs;
  preFFTRep *despoly, *srcpoly;
  TriSet * rests;
  rests=(TriSet *)my_malloc(sizeof(TriSet));
 
  // tsnum has the same bounds as tsden.

  dgs=(sfixn *)my_calloc(N, sizeof(sfixn));
  for(i=0; i<N; i++){
    dgs[i]=BDSI(ts, i+1)+1;
  }
  // going to make the first poly in rests to be Y.
  dgs[0]=1;
 
  initDenseTriSet( N, dgs, rests,  pPtr);

  //pt=1;

  for(i=2; i<=N; i++){
    despoly=ELEMI(rests, i);
    srcpoly=ELEMI(ts, i);
    EvalPolyAtZeroForSmallestVar(despoly, srcpoly, pt, pPtr->P);    
  }
  my_free(dgs);

  // put Y into the new ts.
  DAT(ELEMI(rests, 1))[1]=1;
  DAT(ELEMI(rests, 1))[0]=(pPtr->P)-pt;
  return rests;
}






// The var-list order of the input has ben C-levelized. 
// 
// INPUT: Y:  is the variable has been specialized.
//        y0: is the change of cooridnates.
//        vec_slg: the list of input polynomials.
//        ts: the triangular set.
//        N: # of variables.
// OUTPUT: return the "Lifted,rational reconstructed,remove denomiators" Triangular Set.
//       

/**
 * UniNewtonLift:
 * @iterAddr: a pointer into a sfixn number to fetch the number of interation 
 * has been conducted in the lifting process.
 * @Y: is the var has been specialized.
 * @y0: is the evaluation point had been used to replace 'Y' for intput polynomial.
 * @vec_slg: the input polynomial system (a vector DAGs).
 * @ts: a triangular set.
 * @N: the number of variables.
 * @pPtr: the information of the prime number. 
 *
 * Return value: the lifted trinagular set with rational function coefficients
 **/
TriSet *EX_UniNewtonLift(int *iterAddr, sfixn Y, sfixn y0, POLYVECTOR_SLG * vec_slg, TriSet * ts, sfixn N, MONTP_OPT2_AS_GENE * pPtr){
  TriSet *copyts, *tmpts, *rests;
  RFuncTriSet * rfts;
  POLYVECTOR_SLG *new_vec_slg;
  int consist=1, invertibility;


  // since UniNewtonLift() will distroid inut ts, and turns it to output ts.
  copyts=EX_CopyOneTriSet(ts);

  //printf("before MonicizeTriset_1\n");
  //printf("copyts=");
  //printTriSet(copyts);
  //fflush(stdout);

  invertibility=MonicizeTriSet_1(N, copyts, pPtr);


  //printf("after MonicizeTriset_1\n");
  //printf("copyts=");
  //printTriSet(copyts);
  //fflush(stdout);

  if (invertibility == -1){
          printf("Can't be monicized!\n");
          fflush(stdout);
          freeTriSet(copyts);
          return NULL;
  }


  //  mapped Y to the least variable.
  map_Y2one_1(vec_slg, N, Y);



  //printf("before input system consistent\n");
  //fflush(stdout);

  consist = isInputSystemConsistent(N, vec_slg,  copyts, pPtr, y0);

  //printf("after input system consistent\n");
  //fflush(stdout);



  if(consist == 0) {
    printf("\n\n\n================================================================>\n");
    printf("After making it Lazard Triangular Set, the Input is NOT consistent!\n");
    printf("===================================================================>\n\n\n");
    fflush(stdout);
    EX_freeTriSet(copyts);
    return NULL;
  }
  else{
    // printf("Input System is consistent! continue ...\n");
    // fflush(stdout);
  }


  //subs(y = y + y0, vec_slg); 
  new_vec_slg = shiftInputSystem(vec_slg, 1, y0, pPtr);   
  

  //printf("before UniNewtonlift()\n");
  //fflush(stdout);

  rfts=UniNewtonLift(iterAddr, Y, new_vec_slg, copyts, N, pPtr);

  //printf("after UniNewtonlift()\n");
  //fflush(stdout);


  if(rfts==NULL){
    freeVec_SLG(new_vec_slg);
    EX_freeTriSet(copyts);
    return NULL;     
  }

  freeVec_SLG(new_vec_slg); 

  EX_freeTriSet(copyts);

  tmpts = RemoveDenominators(rfts, pPtr);


  EX_freeTriSet(rfts->numeris);
  EX_freeTriSet(rfts->denomis);

  // subs(y = y - y0, tmpts);
  rests = shiftTriSet(tmpts, (pPtr->P) - y0, pPtr);

  EX_freeTriSet(tmpts);
  
  return rests;


}






