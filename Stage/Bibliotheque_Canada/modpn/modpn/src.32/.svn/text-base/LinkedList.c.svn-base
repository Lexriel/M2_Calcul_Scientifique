/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "LinkedList.h"


/**
 * EX_RegularPair_Init:
 * @poly: polynomial in FFT-representation
 * @ts: (zero-dimensional) triangular set
 *
 * Creates a RegularPair from its members
 * Return value: 
 **/
RegularPair *EX_RegularPair_Init(preFFTRep * poly, TriSet *ts){
  RegularPair *pair = (RegularPair *) my_malloc(sizeof(RegularPair));
    pair->poly = poly;
    pair->ts = ts;
    return pair;}

/**
 * CopyRegularPair:
 * @pair: assumed to be a 'RegularPair'.
 * 
 * Returns a deep copy of `pair`.
 * 
 * Return value: 
 **/
void *EX_CopyRegularPair(void *pair){
  RegularPair *inpair, *newpair;
  inpair= (RegularPair *)pair;
  if(inpair == NULL) return pair;
  newpair = (RegularPair *) my_malloc(sizeof(RegularPair));
  newpair->poly =  EX_CopyOnePoly(inpair->poly);
  newpair->ts =  EX_CopyOneTriSet(inpair->ts);
  return newpair;}

/**
 * EX_RegularPair_Print:
 * @element: assumed to be a 'RegularPair'.
 *
 * Prints the members of `element`.
 * 
 * Return value: 
 **/
void EX_RegularPair_Print(void *element){
  RegularPair *pair = (RegularPair *)element;
  if(pair == NULL){
    printf("NULL.");
  }
  else{
    printf("\n{");
    printPoly(pair->poly);
    printf(".\n");
    printTriSet(pair->ts);
    printf("}\n");
  }
}

/**
 * EX_RegularPair_Free:
 * @element: assumed to be a 'RegularPair'.
 *
 * Frees `element`.
 * 
 * Return value: 
 **/
void EX_RegularPair_Free(void *element){
  RegularPair *pair = (RegularPair *)element;
  if(pair != NULL){
    if(pair->poly  != NULL) EX_FreeOnePoly(pair->poly);
    if(pair->ts != NULL) EX_freeTriSet(pair->ts);
    my_free(pair);
  }}


/**
 * EX_RegularPair_List_Free:
 * @element: A 'RegularPair'
 *  
 * Frees the polynomial of `element` but NOT the triangular set part.
 * 
 * Return value: 
 **/
void EX_RegularPair_List_Free(void *element){
  RegularPair *pair = (RegularPair *)element;
  if(pair != NULL){
    if(pair->poly  != NULL) EX_FreeOnePoly(pair->poly);
    //if(pair->ts != NULL) EX_freeTriSet(pair->ts);
    my_free(pair);
  }}


/**
 * EX_RegularListPair_Init:
 * @no: number of polynomials 
 * @polyList: an array of polynomials
 * @ts: a triangular set
 *
 * Creates the corresponding 'RegularListPair' from its members.
 * 
 * Return value: 
 **/
RegularListPair *EX_RegularListPair_Init(sfixn no, preFFTRep **polyList, TriSet *ts){
  RegularListPair *pair = (RegularListPair *) my_malloc(sizeof(RegularListPair));
    pair->no = no;
    pair->polyList = polyList;
    pair->ts = ts;
    return pair;}


/**
 * EX_CopyRegularListPair:
 * @pair: a 'RegularListPair'
 *
 * Returns a deep of `pair`.
 * 
 * Return value: 
 **/
void *EX_CopyRegularListPair(void *pair){
  RegularListPair *inpair, *newpair;
  int i;
  inpair= (RegularListPair *)pair;
  if(inpair == NULL) return pair;
  newpair = (RegularListPair *) my_malloc(sizeof(RegularListPair));
  newpair->no = inpair->no;
  newpair->polyList = (preFFTRep **) my_malloc((newpair->no) * sizeof(preFFTRep *));
  for(i=0; i<newpair->no; i++){
    (newpair->polyList)[i]=EX_CopyOnePoly((inpair->polyList)[i]);
  }
  newpair->ts =  EX_CopyOneTriSet(inpair->ts);
  return newpair;}


/**
 * EX_RegularListPair_Print:
 * @element: RegularListPair
 * 
 * Prints `element`
 * 
 * Return value: 
 **/
void EX_RegularListPair_Print(void *element){
  RegularListPair *pair = (RegularListPair *)element;
  int i;
  if(pair == NULL){
    printf("NULL.");
  }
  else{
    printf("{");
    for(i=0; i<pair->no; i++){
      printf("poly[%d] = ", i);
      printPoly((pair->polyList)[i]);}
    printf(". ");
    printTriSet(pair->ts);
    printf("}");
  }}



/**
 * EX_RegularListPair_Free:
 * @element: RegularListPair
 * 
 * Frees the whole `element`.
 * 
 * Return value: 
 **/
void EX_RegularListPair_Free(void *element){
  int i;
  RegularListPair *pair = (RegularListPair *)element;
  if(pair != NULL){
    if(pair->polyList  != NULL){
      for(i=0; i< pair->no; i++){
	EX_FreeOnePoly((pair->polyList)[i]);
      }
      my_free(pair->polyList);
    }
    if(pair->ts != NULL) EX_freeTriSet(pair->ts);
    my_free(pair);
  }
}


/**
 * EX_RegularListPair_List_Free:
 * @element: RegularListPair
 * 
 * Frees the polynomials of `element` without freeing its triangular set.
 * 
 * Return value: 
 **/
void EX_RegularListPair_List_Free(void *element){
  int i;
  RegularListPair *pair = (RegularListPair *)element;
  if(pair != NULL){
    if(pair->polyList  != NULL){
      for(i=0; i< pair->no; i++){
	EX_FreeOnePoly((pair->polyList)[i]);
      }
      my_free(pair->polyList);
    }
    //if(pair->ts != NULL) EX_freeTriSet(pair->ts);
    my_free(pair);
  }
}




/**
 * EX_Poly_Print:
 * @element: polynomial (in FFT rep)
 * 
 * Prints `element`.
 * 
 * Return value: 
 **/
void EX_Poly_Print(void *element){
  preFFTRep *poly = (preFFTRep *)element;
  printPoly(poly);
}

/**
 * EX_CopyPoly:
 * @element: polynomial (in FFT rep)
 *  
 * Returns a deep copy of `element`.
 * 
 * Return value: 
 **/
void *EX_CopyPoly(void *element){
  preFFTRep *newpoly, *poly = (preFFTRep *)element;
  if (poly == NULL) return element;
  newpoly = EX_CopyOnePoly(poly);
  return newpoly;
}


/**
 * EX_Poly_Free:
 * @element: polynomial (in FFT rep)
 *  
 * Frees `element`.
 * 
 * Return value: 
 **/
void EX_Poly_Free(void *element){
  preFFTRep *poly = (preFFTRep *)element;
  EX_FreeOnePoly(poly);
}


/**
 * EX_TaskPair2_Init:
 * @i : int
 * @d : int
 * @ts: triangular set
 *
 * Returns a task consisting of an index pair (i, d)
 * and a triangular set.
 * 
 * Return value: 
 **/
TaskPair2 *EX_TaskPair2_Init(int i, int d, TriSet *ts){
  TaskPair2 *pair = (TaskPair2 *) my_malloc(sizeof(TaskPair2));
    pair->i = i;
    pair->d = d;
    pair->ts = ts;
    return pair;
}

/**
 * EX_TaskPair_Init:
 * @index: int
 * @ts: triangular set
 *
 * Returns a task consisting of an index (typically in a subresultant chain)
 * and a triangular set.
 * 
 * Return value: 
 **/
TaskPair *EX_TaskPair_Init(int index, TriSet *ts){
  TaskPair *pair = (TaskPair *) my_malloc(sizeof(TaskPair));
    pair->index = index;
    pair->ts = ts;
    return pair;
}

/**
 * EX_TaskPair2_Print:
 * @element: TaskPair2
 * 
 * Prints `element`.
 * 
 * Return value: 
 **/
void EX_TaskPair2_Print(void *element){

  TaskPair2 *pair = (TaskPair2 *)element;
  if(pair == NULL){
    printf("NULL.");
  }
  else{
    printf("{");
    printf("%d, %d", pair->i, pair->d);
    printf(". ");
    printTriSet(pair->ts);
    printf("}");
  }
}

/**
 * EX_TaskPair_Print:
 * @element: TaskPai
 * 
 * Prints `element`.
 * 
 * Return value: 
 **/
void EX_TaskPair_Print(void *element){

  TaskPair *pair = (TaskPair *)element;
  if(pair == NULL){
    printf("NULL.");
  }
  else{
    printf("{\n");
    printf("index = %d\n", pair->index);
    printf("ts = ");
    printTriSet(pair->ts);
    printf("}");
  }
}

/**
 * EX_TaskPair2_Free:
 * @element: TaskPair2
 * 
 * Frees `element`.
 * 
 * Return value: 
 **/
void EX_TaskPair2_Free(void *element){
  TaskPair2 *pair = (TaskPair2 *)element;
  if(pair != NULL){
    if(pair->ts != NULL) EX_freeTriSet(pair->ts);
    my_free(pair);
  }
}

/**
 * EX_TaskPair_Free:
 * @element: TaskPair
 * 
 * Frees `element`.
 * 
 * Return value: 
 **/
void EX_TaskPair_Free(void *element){
  TaskPair *pair = (TaskPair *)element;
  if(pair != NULL){
    if(pair->ts != NULL) EX_freeTriSet(pair->ts);
    my_free(pair);
  }
}

/**
 * EX_LinearNode_Init:
 * @element: void*
 * 
 * Returns a node whose data member is `element` and link member is 'NULL'
 * 
 * Return value: 
 **/
LinearNode *EX_LinearNode_Init(void *element){
  LinearNode * node = (LinearNode *) my_malloc(sizeof(LinearNode)); 
  node->element = element;
  node->next = NULL;
  return node;
}



/**
 * EX_LinearNode_Print:
 * @node: a node in a linked list (of unspecified type).
 * @printElement: a pointer to a printing function (of unspecified type).
 * 
 * Prints `node`.
 * 
 * Return value: 
 **/
void EX_LinearNode_Print(LinearNode *node, void (*printElement)(void *)){
  if(node != NULL){
    printElement((void *)(node->element));
    printf(" -> ");
  }
  else {
    printf("NULL - >");
  }
}

/**
 * EX_LinearNode_Free:
 * @node: a node in a linked list (of unspecified type).
 * @freeElement: a pointer to a freeing function (of unspecified type).
 * 
 * Frees `node`.
 * 
 * Return value: 
 **/
void EX_LinearNode_Free(LinearNode *node, void (*freeElement)(void *)){
  if(node != NULL){
    if(node->element != NULL)  freeElement(node->element);
    my_free(node);
  }
}

/**
 * EX_LinkedQueue_Ini:
 *
 * Initializes a queue where count gives the number of nodes in the queue.
 * The 'front' and 'rear' point respectively to the first and last nodes.
 * 
 * Return value: 
 **/
LinkedQueue *EX_LinkedQueue_Init(){
  LinkedQueue * LQueue = (LinkedQueue *) my_malloc(sizeof(LinkedQueue));
  LQueue->count = 0;
  LQueue->front = NULL;
  LQueue->rear  = NULL;
  return LQueue;
}

/**
 * EX_LinkedQueue_IsEmpty:
 * @queue: a linkedqueue
 * 
 * Returns 'true' whenever `queue`.
 * 
 * Return value: 
 **/
int EX_LinkedQueue_IsEmpty(LinkedQueue *queue){
  return ((queue->count) == 0);
}

/**
 * EX_LinkedQueue_Print:
 * @queue: a linkedqueue
 * @printElement: a pointer to a printing function
 * 
 * Prints `queue`.
 * 
 * Return value: 
 **/
void EX_LinkedQueue_Print(LinkedQueue *queue, void (*printELement)(void *)){
  LinearNode *current;
  if(! EX_LinkedQueue_IsEmpty(queue)){
    current = queue->front;
    while( current != NULL){
      EX_LinearNode_Print(current, printELement);
      current=current->next;
    }
    printf("\n");
  }
  else{
    printf("Empty Queue!\n");
  }
}



LinkedQueue *
EX_LinkedQueue_Copy(LinkedQueue *queue, void *(*copyElement)(void *) ){
  LinearNode *current;
  LinkedQueue *resQ = EX_LinkedQueue_Init();
  if(! EX_LinkedQueue_IsEmpty(queue)){
    current = queue->front;
    while( current != NULL){
      EX_LinkedQueue_Enqeue(resQ, copyElement((void *)(current->element)));
      current=current->next;
    }
  }
  return resQ;
}



/**
 * EX_LinkedQueue_Free:
 * @queue: a linkedqueue
 * @freeElement: a pointer to a printing function
 * 
 * Frees `queue`.
 * 
 * Return value: 
 **/
void EX_LinkedQueue_Free(LinkedQueue *queue, void (*freeElement)(void *)){
  void *element;
  if(queue != NULL){
    while(! EX_LinkedQueue_IsEmpty(queue)){
      element = EX_LinkedQueue_Deqeue(queue);
      freeElement(element);
    }
    my_free(queue);
  }
}

/**
 * EX_LinkedQueue_Enqeue:
 * @queue: LinkedQueue
 * @element: void*
 * 
 * Appends `element` to `queue`.
 * 
 * Return value: 
 **/
void EX_LinkedQueue_Enqeue(LinkedQueue *queue, void *element){
  LinearNode * node =  EX_LinearNode_Init(element);
  if (EX_LinkedQueue_IsEmpty(queue)){queue->front = node;}
  else{ queue->rear->next = node;}
  queue->rear = node;
  (queue->count)++;
}

/**
 * EX_LinkedQueue_Deqeu:
 * @queue: 
 * 
 * Pops an element (from the front of `queue`).
 * 
 * Return value: 
 **/
void *EX_LinkedQueue_Deqeue(LinkedQueue *queue){
   void *element;
   LinearNode *tmpNode;
   if (EX_LinkedQueue_IsEmpty(queue)){
     printf("The queue is empty!\n"); fflush(stdout);
     Throw 124;
   }

   element = queue->front->element;
   tmpNode=queue->front;
   queue->front = queue->front->next;
   my_free(tmpNode);
   (queue->count)--;
   if (EX_LinkedQueue_IsEmpty(queue)) queue->rear = NULL;
   return element;
}

int EX_LinkedQueue_Size(LinkedQueue *queue){
  return queue->count;
}

/**
 * EX_LinkedQueue_Concat_1:
 * @queue1: 
 * @queue2: 
 * 
 * Appends queue2 to queue1.
 * 
 * Return value: 
 **/
LinkedQueue * EX_LinkedQueue_Concat_1(LinkedQueue *queue1, LinkedQueue *queue2){

  if (EX_LinkedQueue_IsEmpty(queue2)){ return(queue1);}
  if (EX_LinkedQueue_IsEmpty(queue1)==1){
     queue1->front = queue2->front;
  }else{
     queue1->rear->next = queue2->front;
  }
  queue1->rear  = queue2->rear;
  (queue1->count) += (queue2->count);
  return queue1;
 }

/**
 * LinkedQueue2Array:
 * @queue: 
 * @copyElement: 
 * 
 * Converts a linked list to an array with deep copy.
 * 
 * Return value: 
 **/
void **LinkedQueue2Array(LinkedQueue *queue, void *(*copyElement)(void *))
{
  int i=0;
  LinearNode *current;
  void **array = (void **) my_malloc((queue->count)*sizeof(void *));
  current = queue->front;
  for(i=0; i<queue->count; i++){
    array[i] = copyElement((void *)(current->element));
    current=current->next;
  }
  return array;
}
