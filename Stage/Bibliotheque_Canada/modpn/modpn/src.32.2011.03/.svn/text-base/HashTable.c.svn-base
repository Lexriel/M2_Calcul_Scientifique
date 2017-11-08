/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "HashTable.h"

extern int Interrupted;

/**
 * list_create:
 * @opnd: A node value.
 *  
 * Creat a new node has value of 'opnd'.
 * Return value: The new node. 
 **/
NODE *list_create(operand opnd)
{
	NODE *node;
	if(!(node=(NODE *)my_malloc(sizeof(NODE)))) return NULL;
	node->opnd=opnd;
	node->next=NULL;
	return node;
}


/**
 * list_insert_after:
 * @node: A node.
 * @opnd: A node value.
 *  
 * Creat a new node has value of 'opnd', and append the new node onto 'node'.
 * Return value: The new node. 
 **/
NODE *list_insert_after(NODE *node, operand opnd)
{

	NODE *newnode;
        newnode=list_create(opnd);
        newnode->next = node->next;
        node->next = newnode;
	return newnode;


}


/**
 * list_insert_beginning:
 * @list: A linked list.
 * @opnd: A node value.
 *  
 * Creat a new node has value of 'opnd', and push onto the head of 'list'.
 * Return value: The new node. 
 **/
NODE *list_insert_beginning(NODE *list, operand opnd)
{
	NODE *newnode;
        newnode=list_create(opnd);
        newnode->next = list;
	return newnode;
}



/**
 * list_remove:
 * @list: A linked list.
 * @node: A node.
 *  Remove the 'node' from the 'list'.
 * Return value: 0 if node is found and removed. otherwise -1.
 **/
int list_remove(NODE *list, NODE *node)
{
	while(list->next && list->next!=node) list=list->next;
	if(list->next) {
		list->next=node->next;
		my_free(node);
		return 0;		
	} else return -1;
}


// equal -> return 1;
// otherwise -> return 0;

/**
 * compareOperand:
 * @opnd1: Value of a node.
 * @opnd2: Value of a node.
 * 
 * Comparing if two values are equal.
 * Return value: if the input are equal returns 1, otherwise returns 0. 
 **/
int 
compareOperand(operand opnd1, operand opnd2){

  if ( type_of(opnd1) != type_of(opnd2)) return 0;

     switch (type_of(opnd1)) {
	case t_poly:
          printf("Error: not supposed to be poly type! in function compareTriple()\n");
          fflush(stdout);
          Interrupted=1;
	  return 0;
	case t_sfixn:
          if(sfixn_val(opnd1) == sfixn_val(opnd1)) return 1;
          else return 0;
	case t_var:
          if(var_no(opnd1) == var_no(opnd2)) return 1;
          else return 0;
	case t_varpow:
          printf("varPow to be tested or implemented. \n");
	  Throw 104;      
	case t_biPlus:
          if(      ((biPlus_oper1(opnd1)==biPlus_oper1(opnd2)) 
		    && (biPlus_oper2(opnd1)==biPlus_oper2(opnd2)))
              || 
                   ((biPlus_oper1(opnd1)==biPlus_oper2(opnd2)) 
		    && (biPlus_oper2(opnd1)==biPlus_oper1(opnd2)))
            ) return 1;
          else return 0;
	case t_biSub:
          if(      ((biSub_oper1(opnd1)==biSub_oper1(opnd2)) 
		    && (biSub_oper2(opnd1)==biSub_oper2(opnd2)))
            ) return 1;
          else return 0;
	case t_biProd:
         if(      ((biProd_oper1(opnd1)==biProd_oper1(opnd2)) 
		    && (biProd_oper2(opnd1)==biProd_oper2(opnd2)))
              || 
                   ((biProd_oper1(opnd1)==biProd_oper2(opnd2)) 
		    && (biProd_oper2(opnd1)==biProd_oper1(opnd2)))
            ) return 1;
          else return 0;      
	default:
          printf("Error: No this type in function compareTriple()\n");
	  fflush(stdout);
          Interrupted=1;
	}

}



/**
 * list_find:
 * @head: The head of a linked list.
 * @opnd: A node value.
 * 
 * Search a linked list starting from the 'head'.
 * If a node has value "opnd", then return this node, otherwise return NULL.
 * Return value: A node has the value "opnd".
 **/
NODE *list_find(NODE *head, operand opnd)
{
  
	while(head) {
	  if(compareOperand(head->opnd, opnd)) return head;
	     head=head->next;
	}
	return NULL;
}



/**
 * list_free:
 * @list: A linked list.
 * Free a linked list.
 * Return value: 
 **/
void list_free(NODE *list)
{  NODE *head, *tmp;
   head=list;
   while(head != NULL){
     tmp=head;
     head=head->next;
     my_free(tmp);
   }
}







/**
 * hashFunction:
 * @opnd: input value. 
 * 
 * Return value: the Hash sum. 
 **/
int hashFunction(operand opnd){

       switch (type_of(opnd)) {
	case t_poly:
          printf("not supposed to be poly type! in function compareTriple()\n");
	  Throw 104;
	case t_sfixn:
          return (HashModuli + sfixn_val(opnd) % HashModuli) % HashModuli;
	case t_var:
          return var_no(opnd) % HashModuli;
	case t_varpow:
          printf("varPow to be tested or implemented. \n");
	  Throw 104;      
	case t_biPlus:
          return ((int)t_biPlus + (int)(ulongfixnum)(biPlus_oper1(opnd)) + 
                  (int)(ulongfixnum)(biPlus_oper2(opnd)))% (int)HashModuli;

	case t_biSub:
          return ((int)t_biSub + (int)(ulongfixnum)(biSub_oper1(opnd)) + 
                  (int)(ulongfixnum)(biSub_oper2(opnd)))% HashModuli;
	case t_biProd:
          return ((int)t_biProd + (int)(ulongfixnum)(biProd_oper1(opnd)) + 
                  (int)(ulongfixnum)(biProd_oper2(opnd)))% (int)HashModuli;  
	default:
          printf("Error: No this type in function compareTriple()\n");
	  Throw 105;
	}

}

// yes, return the existing node.
// no, return a NULL;

/**
 * searchNodeinHashTable:
 * @opnd: The research key.
 * @hash_table: The Hash table.
 *   Search the Hash table to find the node has the same researh key.
 * Return value: The node has the same input research key. 
 **/
operand
searchNodeinHashTable(operand opnd, HASHTABLE *hash_table){
  NODE * theList;
  operand result;
  theList = (hash_table->table)[hashFunction(opnd)];
  //if(theList == NULL) printf("theList is NULL \n");
  theList = list_find(theList, opnd);
  if(theList) result = theList->opnd; else result = (operand)NULL;
  //if(result == NULL) printf("result is NULL! \n");
  //fflush(stdout);
  return result;
}



/**
 * newHashTable:
 * @length: a integer.
 *  Create a new Hash table with 'length' slots.
 * Return value: The new Hash table. 
 **/
HASHTABLE *
newHashTable(unsigned int length){
  HASHTABLE *hash_table = (HASHTABLE *)my_malloc(sizeof(HASHTABLE));
  hash_table->length = length;
  hash_table->table = (NODE **)my_calloc(length, sizeof(NODE *));
  //printf("hash_table->length = %d. \n", length);
  //fflush(stdout);
  return hash_table;
}


/**
 * freeHashTable:
 * @hash_table: a Hash table.
 *  Free a hash table.
 * Return value: 
 **/
void
freeHashTable(HASHTABLE *hash_table){
  int i;
  for(i=0; i<(hash_table->length); i++){
    if((hash_table->table)[i]) list_free((hash_table->table)[i]);
  }
  if(hash_table->table) my_free(hash_table->table);
  my_free(hash_table);
}
