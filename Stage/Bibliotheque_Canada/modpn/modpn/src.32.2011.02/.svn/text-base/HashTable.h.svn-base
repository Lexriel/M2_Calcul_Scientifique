/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#ifndef __HashTable_h
#define __HashTable_h 

#include "Types.h"
#include "UniHensel.h"

#define  HashModuli 101

typedef struct node_struct NODE; 
typedef struct hashtable_struct HASHTABLE;

struct node_struct {
  operand opnd;
  struct node_struct *next;	
};


// array index is the key value.
struct hashtable_struct
{
        NODE ** table;
	unsigned int length;
};


NODE *list_create(operand opnd);

NODE *list_insert_after(NODE *node, operand opnd);

NODE *list_insert_beginning(NODE *list, operand opnd);

int list_remove(NODE *list, NODE *node);

int compareOperand(operand opnd1, operand opnd2);

NODE *list_find(NODE *node, operand opnd);

void list_free(NODE *list);

int hashFunction(operand opnd);


// yes, return the existing node.
// no, return a NULL;
operand searchNodeinHashTable(operand opnd, HASHTABLE *hash_table);



HASHTABLE *newHashTable(unsigned int length);


void freeHashTable(HASHTABLE *hash_table);


operand  searchNodeinHashTable(operand opnd, HASHTABLE *hash_table);

#endif
