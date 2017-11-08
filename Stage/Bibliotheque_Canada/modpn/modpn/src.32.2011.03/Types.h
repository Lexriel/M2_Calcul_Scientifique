/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#ifndef __Types_h
#define __Types_h 

#define WINDOWS 1

#define DEBUG 0
#define DEBUG1 0
#define DEBUG2 0
#define DEBUG3 0
#define DEBUG4 0
#define DEBUG5 0
#define DEBUG20 0
#define DEBUG21 0
#define DEBUG22 0
#define DEBUG23 0
#define DEBUG24 0
#define DEBUG25 0
#define DEBUG30 0
#define DEBUG31 0
#define DEBUG33 0
#define DEBUG35 0
#define DEBUG38 0
#define DEBUG39 0
#define NoOfCPU 4

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cexcept.h"
#include <signal.h>

#ifdef _MSC_VER
#define CDECL __cdecl
#else
#define CDECL
#endif

/* Cuda enabled architectures */
//#if defined(WINDOWS) || defined(LINUXINTEL32) || defined(LINUXINTEL64)
#if defined(LINUXINTEL64)
#define _cuda_modp_
#endif

define_exception_type(int);
extern struct exception_context the_exception_context[1];

#define _mcompile_

/* modpn and cumodp integer types */
#ifndef __modpn_integer_types__
#define __modpn_integer_types__

#ifdef LINUXINTEL32
#undef WINDOWS
typedef long               sfixn;
typedef long long          longfixnum;
typedef unsigned long      usfixn;
typedef unsigned long long ulongfixnum;
typedef long               int32;
typedef unsigned long      uint32;
#endif

#ifdef LINUXINTEL64
#undef WINDOWS
typedef int            sfixn;
typedef long           longfixnum;
typedef unsigned int   usfixn;
typedef unsigned long  ulongfixnum;
typedef int            int32;
typedef unsigned int   uint32;
#endif

#ifdef MAC32
#undef WINDOWS
#include <stdint.h>
typedef int32_t   sfixn;
typedef int64_t   longfixnum;
typedef uint32_t  usfixn;
typedef uint64_t  ulongfixnum;
typedef int32_t   int32;
typedef uint32_t  uint32;
#endif

#ifdef SOLARIS64
#undef WINDOWS
typedef int32_t   sfixn;
typedef int64_t   longfixnum;
typedef uint32_t  usfixn;
typedef uint64_t  ulongfixnum;
typedef int32_t   int32;
typedef uint32_t  uint32;
#endif

#ifdef PPC64
#undef WINDOWS
typedef int32_t   sfixn;
typedef int64_t   longfixnum;
typedef uint32_t  usfixn;
typedef uint64_t  ulongfixnum;
typedef int32_t   int32;
typedef uint32_t  uint32;
#endif

#ifdef MAC64
#undef WINDOWS
typedef int32_t   sfixn;
typedef int64_t   longfixnum;
typedef uint32_t  usfixn;
typedef uint64_t  ulongfixnum;
typedef int32_t   int32;
typedef uint32_t  uint32;
#endif

#ifdef WINDOWS
typedef _int32           sfixn;
typedef _int64           longfixnum;
typedef unsigned _int32  usfixn;
typedef unsigned _int64  ulongfixnum;
typedef _int32           int32;
typedef unsigned _uint32 uint32;
#endif

#endif /* END OF INTEGER TYPES */

/**
 * preFFTRep: "cube" representation of multivariate dense  
 * @N: number of variables = number of dimensions in the cube
 * @defN: backup of N 
 * @bufSizs: 1-dim array of length N+1.
 *           bufSizs[i]+1 is the size of the buffer at dimension i.
 *           bufSizs[0] is not used.
 * @cuts:   1-dim array of length N+1. Gives the partial degrees.
 * @accum:  1-dim array of length N+1.
 *          accum[i] is the size of a coefficient w.r.t. the i-th variable.
 * @size:   the size of the coefficient buffer.
 * @defSize:  backup of size
 * @offset: shold be zero
 * @data: coefficient buffer
 * @tmpData: backup for the data pointer
 * @defData: backup for the data pointer
 * 
 * The elements of the cube are coefficients (not values).
 * 
 * Return value: 
 **/   
typedef struct preFFTRepST{
  sfixn N; 
  sfixn defN;
  sfixn * bufSizs; // bufSizes is a erroric name. It should be degrees.
                   // I.e. bufSizs is a vector keeps the partial degrees of 
                   // the give polynomias
  sfixn * cuts;  
  sfixn * accum;
  sfixn size;  
  sfixn datasize;
  sfixn defSize;
  sfixn offset;
  sfixn * data;
  sfixn * tmpData; 
  sfixn * defData; 
} preFFTRep;

#define N(Ptr)              (Ptr->N)
#define DFN(Ptr)            (Ptr->defN)
#define BUSZSI(Ptr, i)      ((Ptr->bufSizs)[i])
#define BUSZS(Ptr)          (Ptr->bufSizs)
#define BUSZSD(Ptr)         (Ptr.bufSizs)
#define BUSZSDI(Ptr, i)     ((Ptr.bufSizs)[i])
#define CUTSI(Ptr, i)       ((Ptr->cuts)[i])
#define CUTS(Ptr)           (Ptr->cuts)
#define CUTSD(Ptr)          (Ptr.cuts)
#define CUTSDI(Ptr, i)      ((Ptr.cuts)[i])
#define CUMI(Ptr, i)        ((Ptr->accum)[i])
#define CUM(Ptr)            (Ptr->accum)
#define CUMD(Ptr)           (Ptr.accum)
#define SIZ(Ptr)            (Ptr->size)
#define DFSIZ(Ptr)          (Ptr->defSize)
#define OFST(Ptr)           (Ptr->offset)
#define DATI(Ptr, i)        ((Ptr->data)[i])
#define DAT(Ptr)            (Ptr->data)
#define DATD(Ptr)           (Ptr.data)
#define TMPDAT(Ptr)         (Ptr->tmpData)
#define DEFDAT(Ptr)         (Ptr->defData)

/**
 * KroFFTRep: 
 * @N: number of variables
 * @M: number of polynomias to be multiplied (must be 2 most of the time)
 * @es: 2^(es[i]) is the FFT size on dimension i 
 * @dims: dims[i]  FFT size on dimension i 
 * @size: size of the Kronecker encoding of any of the input polynomials
 * @Defsize: backup of size
 * @accum: 1-dim array of length N+1.
 *          accum[i] is the size of a coefficient w.r.t. 
 *          the i-th variable  of the product.
 * @datas:  array of pointers to the datas of the Kronecker encoding of the
 *          input polynomials. datas[0] and datas[1] point to the first
 *          and second data encodings.
 * @KN: FFT-size of the univariate product
 * @KE: 2^KE = KN
 * @KrootsPtr: powers of the primitive root of unity used for this produc
 * 
 * Return value: 
 **/   
typedef struct KroFFTRepST{
  sfixn N; 
  sfixn M; 
  sfixn * es; 
  sfixn * dims; 
  sfixn size; 
  sfixn Defsize; 
  sfixn * accum; 
  sfixn ** datas; 
  sfixn KN; 
  sfixn KE; 
  sfixn * KrootsPtr; 
} KroFFTRep;

#define M(Ptr)              (Ptr->M)
#define ES(Ptr)             (Ptr->es)
#define ESI(Ptr, i)         ((Ptr->es)[i])
#define DIMS(Ptr)           (Ptr->dims)
#define DIMSI(Ptr, i)       ((Ptr->dims)[i])
#define DATS(Ptr)           (Ptr->datas)
#define DATSI(Ptr, i)       ((Ptr->datas)[i])
#define KN(Ptr)             (Ptr->KN)
#define KE(Ptr)             (Ptr->KE)
#define KROOTS(Ptr)         (Ptr->KrootsPtr)

/**
 * KroTFTRep: 
 * @N: number of variables
 * @M: number of polynomias to be multiplied (must be 2 most of the time)
 * @es: 2^(es[i]) is the FFT size on dimension i 
 * @dims: dims[i]  FFT size on dimension i 
 * @ls: ls[i] TFT size on dimension i (= FFT size - number-of-leading-zeros) 
 * @size: size of the Kronecker encoding of any of the input polynomials
 * @Defsize: backup of size
 * @accum: 1-dim array of length N+1.
 *          accum[i] is the size of a coefficient w.r.t. 
 *          the i-th variable  of the product.
 * @datas:  array of pointers to the datas of the Kronecker encoding of the
 *          input polynomials. datas[0] and datas[1] point to the first
 *          and second data encodings.
 * @KN: FFT-size of the univariate product
 * @KE: 2^KE = KN
 * @KrootsPtr: powers of the primitive root of unity used for this produc
 * 
 * Return value: 
 **/   
typedef struct KroTFTRepST{
  sfixn N; 
  sfixn M; 
  sfixn * es; 
  sfixn * dims;
  sfixn * ls; 
  sfixn size; 
  sfixn Defsize; 
  sfixn * accum; 
  sfixn ** datas; 
  sfixn KN; 
  sfixn KE; 
  sfixn * KrootsPtr; 
} KroTFTRep;
#define LS(Ptr)           (Ptr->ls)
#define LSI(Ptr, i)       ((Ptr->ls)[i])

/**
 * TriSetST: Data structure for zero-dim regular chains.
 *           Can be used for non zero-dim regular chain BUT this with care!
 *           Implictely we work with variables X1 < ... < XN.
 * @normalized: 1 means yes, 0 means no.
 *              Here normalized means reduced in the sense of Grobner bases
 *              and each initial is ONE.
 * @N:         the number of polynomials
 * @bounds:    bounds[i] + 1 i the degree of the i-th polynomial
 *             w.r.t. variable Xi.
 * @elems:     elems[i] is the polynomial whose main variable is Xi
 *             (if any, otherwise 0).
 * 
 * Return value: 
 **/   
typedef struct TriSetST{
  sfixn normalized;  // 1 means yes, 0 means no.
  sfixn N; 
  sfixn *bounds; 
  preFFTRep **elems; // elems[0] is useless.
}TriSet;
#define NMLZ(Ptr)             (Ptr->normalized)
#define BDS(Ptr)              (Ptr->bounds)
#define BDSI(Ptr, i)          ((Ptr->bounds)[i])
#define ELEM(Ptr)             (Ptr->elems)
#define ELEMI(Ptr, i)         ((Ptr->elems)[i])

/* Pair of triangular sets, only used by lifting */
typedef struct RFuncTriSetST{
  TriSet *numeris;
  TriSet *denomis;
}RFuncTriSet;

/**
 * TriRevInvSet: modular inverses of the elements of zero-dim reg chain
 *               to be used in the fast normal form (based on Newton iteration)
 * @N: number of polynomials = variables
 * @exist:  exist[i]=1  means the RevInv(T_i) is already computed.
 *          exist[0] is undefined
 * @NewtonLbounds: NewtonLbounds[i] is the smallest power of greater or equal
 *                 to the main degree of the  i-th quotient plus one
 * @NewtonSbounds: NewtonSbounds[i] is the degree of the  i-th quotient 
 *                 plus one.
 * @elems:         elems[i] is the inverse if the i-th polynomial in the
 *                 triangular set modulo the i-th variable and the lower
 *                 polynomials; moreover the coefficients w.r.t. 
 *                 the i-th variable are in reverse order.
 * 
 * Return value: 
 **/   
typedef struct TriRevInvSetST{
  sfixn N; // # of polys in this triangular set.
  sfixn * exist; // exist[0] is undefined. exist[i]=1 
                  // means the RevInv(T_i) is already computed.
  sfixn * NewtonLbounds; // power of 2 in Newton iteration.
  sfixn * NewtonSbounds; // real bound in Newton Iteration.
  preFFTRep ** elems; // keeps the inverses.
}TriRevInvSet;
#define EXSTI(Ptr, i)       ((Ptr->exist)[i])
#define EXST(Ptr)           (Ptr->exist)
#define NLB(Ptr)            (Ptr->NewtonLbounds)
#define NSB(Ptr)            (Ptr->NewtonSbounds)

/* Pure Montgommery trick for integers */
typedef struct MONTPRIMESTRUCT{
  sfixn  v;
  sfixn  Rpow;
  sfixn  Rmask;
  sfixn  Rinv;
  sfixn  V;
  sfixn  Vneg;
}  MONTP_GENE;

/* Used for the improved Montgommery trick for integers */
typedef struct MONTPRIMEOPT2STRUCT{
  sfixn  P;
  sfixn  c;
  sfixn  Npow;
  sfixn  Rpow;
  sfixn  R_Npow;
  sfixn  Base_Rpow;// 32-Row on 32-bit machine.
  sfixn  Base_Npow;// 32-Row on 32-bit machine.
  sfixn  c_sft; // only for assembly subroutine.
  sfixn  c_pow; // 2^c_pow+1=c. for special case.
  sfixn  N2_Rpow;
  sfixn  Rmask;
  sfixn  R_Nmask;
  sfixn  Max_Root;
  sfixn  R2BRsft;
}  MONTP_OPT2_AS_GENE;

#ifndef plong
#define plong int
#endif
// Add to fix the lifting allocation problem! Change-Code: lift-0
typedef union operandUnion operandObj;
typedef union operandUnion *operand;
typedef struct SL_Graph SLG;
typedef void *Pointer;

// Macros for the types of nodes in the DAG representation
// of polynomials as used in the lifting
typedef struct dummy_struct DUMYO;
typedef struct sfixn_struct SFIXNO;
typedef struct variable_struct VARO;
typedef struct variablePow_struct VARPOWO;
typedef struct biPlus_struct BIPLUSO;
typedef struct biSub_struct BISUBO;
typedef struct biProd_struct BIPRODO;
typedef struct polynomial_struct POLYO;
typedef struct pow_struct POWO;

#define Ntypes 8

/* Type of a node in a DAG representing a polyomial */
typedef
enum type {
  t_poly,   // 0
  t_sfixn,  // 1
  t_var,    // 2 (1..n  ~  x1 ~ xn)   
  t_varpow, // 3  var^e has NOT been implemented or tested.
  t_biPlus, // 4
  t_biSub,  // 5
  t_biProd, // 6
  t_pow     // 7  operand^e  has NOT been implemented or tested.
}operandType;

// The following are macros for operating on the nodes of a DAG
#define	type_of(oper)	    ((operandType)(((operand)(oper))->DUMY.type))
#define	type_set(oper, t)   ((((operand)(oper))->DUMY.type)=t)
#define	id_of(oper)	        ((operandType)(((operand)(oper))->DUMY.id))
#define	id_set(oper, theid)	((((operand)(oper))->DUMY.id)=theid)
#define is_poly(oper)       (type_of(oper) == t_poly)
#define is_var(oper)        (type_of(oper) == t_var)
#define is_sfixn(oper)      (type_of(oper) == t_sfixn)
#define is_varpow(oper)     (type_of(oper) == t_varpow)
#define is_biPlus(oper)     (type_of(oper) == t_biPlus)
#define is_biSub(oper)      (type_of(oper) == t_biSub)
#define is_biProd(oper)     (type_of(oper) == t_biProd)
#define is_pow(oper)        (type_of(oper) == t_pow)
#define poly_poly(oper)     (((oper)->POLY).poly)
#define sfixn_val(oper)     (((oper)->SFIX).sfixnVal)
#define var_no(oper)        (((oper)->VAR).no)
#define varPow_e(oper)      (((oper)->VARPOW).e)
#define varPow_no(oper)     (((oper)->VARPOW).no)
#define biPlus_oper1(oper)  (((oper)->BI_PLUS).oper1)
#define biPlus_oper2(oper)  (((oper)->BI_PLUS).oper2)
#define biSub_oper1(oper)   (((oper)->BI_SUB).oper1)
#define biSub_oper2(oper)   (((oper)->BI_SUB).oper2)
#define biProd_oper1(oper)  (((oper)->BI_PROD).oper1)
#define biProd_oper2(oper)  (((oper)->BI_PROD).oper2)
#define pow_e(oper)         (((oper)->VARPOW).e)
#define pow_base(oper)      (((oper)->VARPOW).base)

#define is_sameVar(oper1, oper2) \
    (((oper1)->VAR).no == ((oper2)->VAR).no) 

#define is_inSameVar(oper1, oper2) \
    (((oper1)->VARPOW).no == ((oper2)->VAR).no)

#define new_poly(oper) \
    oper = (operand)my_calloc(1, sizeof(POLYO)); \
    type_set(oper, t_poly) 

#define new_poly_ini(oper, thepoly) \
    oper = (operand)my_calloc(1, sizeof(POLYO)); \
    type_set(oper, t_poly); \
    (oper->POLY).poly = thepoly

#define new_sfixn(oper) \
    oper = (operand)my_calloc(1, sizeof(SFIXNO)); \
	type_set(oper, t_sfixn) 

#define new_sfixn_ini(oper, v) \
    oper = (operand)my_calloc(1, sizeof(SFIXNO)); \
    type_set(oper, t_sfixn); \
    (oper->SFIX).sfixnVal = v

#define	new_var(oper) \
    oper = (operand)my_calloc(1, sizeof(VARO)); \
	type_set(oper, t_var)

#define	new_var_ini(oper, newno) \
    oper = (operand)my_calloc(1, sizeof(VARO)); \
    type_set(oper, t_var);\
    (oper->VAR).no = newno

#define	new_varpow(oper) \
    oper = (operand)my_calloc(1, sizeof(VARPOWO)); \
    type_set(oper, t_varpow)

#define	new_varpow_ini(oper, newno, newe) \
    oper = (operand)my_calloc(1, sizeof(VARPOWO)); \
    type_set(oper, t_varpow); \
    (oper->VARPOW).e = newe; \
    (oper->VARPOW).no = newno

#define	new_biPlus(oper) \
    oper = (operand)my_calloc(1, sizeof(BIPLUSO));	\
    type_set(oper, t_biPlus)

#define	new_biPlus_ini(oper, newoper1, newoper2) \
    oper = (operand)my_calloc(1, sizeof(BIPLUSO)); \
    type_set(oper, t_biPlus); \
    (oper->BI_PROD).oper1 = newoper1; \
    (oper->BI_PROD).oper2 = newoper2                             

#define	new_biSub(oper) \
    oper = (operand)my_calloc(1, sizeof(BISUBO));	\
    type_set(oper, t_biSub)

#define	new_biSub_ini(oper, newoper1, newoper2) \
    oper = (operand)my_calloc(1, sizeof(BISUBO)); \
    type_set(oper, t_biSub); \
    (oper->BI_PROD).oper1 = newoper1; \
    (oper->BI_PROD).oper2 = newoper2

#define	new_biProd(oper) \
    oper = (operand)my_calloc(1, sizeof(BIPRODO));	\
    type_set(oper, t_biProd) 

#define	new_biProd_ini(oper, newoper1, newoper2) \
    oper = (operand)my_calloc(1, sizeof(BIPRODO)); \
    type_set(oper, t_biProd); \
    (oper->BI_PROD).oper1 = newoper1; \
    (oper->BI_PROD).oper2 = newoper2

#define	new_pow(oper) \
    oper = (operand)my_calloc(1, sizeof(POWO)); \
    type_set(oper, t_pow)

#define	new_pow_ini(oper, newe, newbase) \
    oper = (operand)my_calloc(1, sizeof(POWO)); \
    type_set(oper, t_pow); \
    (oper->VARPOW).e = newe; \
    (oper->VARPOW).base = newbase

/** 
 * FIRSTWORD is a data-structure storing the following information about 
 * each node in a DAG representing a polynomial.
 *
 * type   -- data type.
 * flag   -- bit-0 tmpMark0 (liveness).
 *           bit-1 tmpMark1 (derivative).
 *           bit-2 tmpMark2 (copy).
 *           bit-3 tmpMark3 (tmp dead mark).
 *           bit-4~7 reserved.
 * id     -- The ID of a object (a node in G.)
 **/
#define FIRSTWORD unsigned char type, flag; unsigned short int id  

#define tmpMarkMask 0x2
#define tmpMark2Mask 0x4
#define tmpMark3Mask 0x8
#define tmpMarkUnMask 0xfd
#define tmpMark2UnMask 0xfb
#define tmpMark3UnMask 0xf7

#define	is_TmpMark1On(oper)	 ((((operand)(oper))->DUMY.flag) & tmpMarkMask)
#define	is_TmpMark2On(oper)	 ((((operand)(oper))->DUMY.flag) & tmpMark2Mask)
#define	is_TmpMark3On(oper)	 ((((operand)(oper))->DUMY.flag) & tmpMark3Mask)
#define	clear_TmpMark1(oper) ((((operand)(oper))->DUMY.flag) &= tmpMarkUnMask)
#define	clear_TmpMark2(oper) ((((operand)(oper))->DUMY.flag) &= tmpMark2UnMask)
#define	clear_TmpMark3(oper) ((((operand)(oper))->DUMY.flag) &= tmpMark3UnMask)
#define	set_TmpMark1(oper)	 ((((operand)(oper))->DUMY.flag) |= tmpMarkMask)
#define	set_TmpMark2(oper)	 ((((operand)(oper))->DUMY.flag) |= tmpMark2Mask)
#define	set_TmpMark3(oper)	 ((((operand)(oper))->DUMY.flag) |= tmpMark3Mask)

/* dummy struct. */
struct dummy_struct {
    FIRSTWORD;
};

/* single fixnum. */
struct sfixn_struct {
    FIRSTWORD;
    sfixn sfixnVal;
};

/* Variable. */
struct variable_struct {
    FIRSTWORD;
    int no; // no=1 -> is x.
};

/* VariablePow. */
struct variablePow_struct {
	FIRSTWORD;
    int e;  // e is the exponent .
    int no; // no=1 -> is x.
};

struct biPlus_struct {
    FIRSTWORD;
    operand oper1;
    operand oper2; // when Uni-operation op2 is NULL.
};

struct biSub_struct {
    FIRSTWORD;
    operand oper1;
    operand oper2; // when Uni-operation op2 is NULL.
};

struct biProd_struct {
	FIRSTWORD;
    operand oper1;
    operand oper2; // when Uni-operation op2 is NULL.
};

struct polynomial_struct {
    FIRSTWORD;
    preFFTRep *poly;    
};

/* pow. */
struct pow_struct {
    FIRSTWORD;
    int e;  // e is the exponent .
    operand base; // no=1 -> is x.
};

// Each of the DUMYO, .., POWO are macros
// for the node types defined above.
// operand types.
union operandUnion {
    DUMYO   DUMY;
    POLYO   POLY;
	SFIXNO  SFIX;
    VARO    VAR;
    VARPOWO VARPOW;
    BIPLUSO BI_PLUS;
    BISUBO  BI_SUB;
    BIPRODO BI_PROD;
    POWO    POW;
};

// A graph G for the Strait Line input.
// The strait line is a graph with following property.
//  0) encoded in the adjacency-list style. 
//  1) G is connected or unconnected.
//  2) A node's children must appear efore this node in the nodes vector.
//  3) the sum of all subgraphs in G is the complete encoding for the give polynomial.
struct SL_Graph{
  int GN;          // Number of Nodes  G
  int GE;          // Number of Edges in G
  operand * Nodes; // a vector of nodes. Typical adjacency-list encoding. 
                   // Nodes has size GN; Nodes[i] is the i-th nodes
                   // and Nodes[i] has ID i, for i=1...GN
};

// macros for operations on the DAG
#define GN(slg)  (slg->GN)
#define GE(slg)  (slg->GE)
#define ROOT_G(slg)  ((slg->Nodes)[(slg->GN) - 1])
#define setGN_G(slg, N)  ((slg->GN)=N)
#define NodeI(slg, i)  ((slg->Nodes)[i])

typedef struct polyMatrix_struct  POLYMATRIX;

// M by N matrix of polynomials
struct polyMatrix_struct{
  int M; // number of rows;
  int N; // number of columns;                  
  preFFTRep **entries;
};

// macros for matrix operations
#define ENTRYI_M(polyM, m, n) ((polyM->entries)[m*(polyM->N) + n])

typedef struct polyVector_SLG_struct  POLYVECTOR_SLG;

// Could be obselete ...
struct polyVector_SLG_struct{
  int M; // #;                  
  SLG **entries;
};

#define ENTRYI_V(polyV, i) ((polyV->entries)[i])

typedef struct polyVector_struct  POLYVECTOR;

// Vector of polynomials
struct polyVector_struct{
  int M; // #;                  
  preFFTRep **entries;
};

typedef struct subProdTree_struct subProdTree;

// level_0 is useless
// level_1 is the bottom level.
// level_{h+1} is the root.
 /**
 * subProdTree: subproduct tree structure for degree one moduli polynomials
 * @h: height of the tree; the root is at level 0
 * @W: 1-dim array of size h+1;  W[n] is the node size of level n. 
 * @NoNodes: 1-dim array of size h+1; NoNodes[n] the # of nodes at level n.
 * @data: the actual data: coefficients for levels 0,1,... consecutively
 * @Bases: 1-dim array of size h+1; Bases[n] is the position of the first
 *         coefficient of the first polynomial at level n
 * 
 * Return value: 
 **/   
struct subProdTree_struct{
  sfixn h;
  sfixn *W; // node of size. W[n] is the node size of level n. 
  sfixn *NoNodes; // NoNodes[n] the # of nodes at level n.
  sfixn *data;
  sfixn *Bases;
};

typedef struct pts_trees_struct  PTS_TREE;

/**
 * PTS_TREE: subprodtree structure for fast evaluation / interpolation
 * @no: number of variables = number of trees
 * @ptsPtr: array of arrays of values (= points)
 *           ptsPtr[i] gives the evaluation points in the i-th dim
 * @trees:  array of the corresponding subprodtrees
 * 
 * Return value: 
 **/   
struct pts_trees_struct{
  sfixn no; // no of trees / <set of points>.
  sfixn **ptsPtr; // has no+1 slots, first one is not used.
  subProdTree **trees; // has no+1 slots, first one is not used.
  sfixn times;
};

/**
 * InterpRFRST: data-structure for rational function reconstruction 
 * @bound: degree of the modulo
 * @degNum:  actual degree of the numerator of the reconstructed fraction
 * @Num: numerator of the reconstructed fraction
 * @degDen: actual degree of the denominator of the reconstructed fraction
 * @Den: denominator of the reconstructed fraction
 * 
 * Return value: 
 **/  
typedef struct InterpRFRStruct{
  sfixn bound;
  sfixn degNum;
  sfixn *Num;
  sfixn degDen;
  sfixn *Den;
} InterpRFRST;

/**
 * InterpRFRPreST:  data-structure for rational function interpolation
 *                  But more precisely for the modular computation of
 *                  iterated resultants modulo a 1-dimreg chain
 * @no: number of points
 * @tree: the corresponding subprodtree 
 * @points: the points 
 * @values: the images of the iterated resultant at the points 
 * @chains: the images of the input regular chain at the points 
 * @polys:   the images of the input polynomial at the points 
 * 
 * Return value: 
 **/   
typedef struct InterpRFRPreStruct{
  sfixn no;  // Number of points.
  subProdTree *tree;
  sfixn *points;
  sfixn *values;
  TriSet **chains;
  preFFTRep **polys; 
} InterpRFRPreST;

// Pair of a polynomial and a triangular set
struct RegularPairST{
  preFFTRep *poly;
  TriSet *ts;
};

typedef struct RegularPairST RegularPair;

// Pair of a polynomial list and a triangular set
struct RegularListPairST{
  sfixn no;
  preFFTRep **polyList;
  TriSet *ts;
};
typedef struct RegularListPairST  RegularListPair;

struct TaskPairST{
  int index;
  TriSet *ts;
};

typedef struct TaskPairST TaskPair;

struct TaskPairST2{
  int i;
  int d;  
  TriSet *ts;
};

typedef struct TaskPairST2 TaskPair2;

// A node in a linked list /  queueu
struct LinearNodeST{
  void *element;
  struct LinearNodeST *next;
};

typedef struct LinearNodeST LinearNode;

struct LinkedQueueST{
  int count;
  LinearNode *front, *rear;
};

typedef struct LinkedQueueST LinkedQueue;

/**
 * Data-structure for computing the subresultant chains of two multivariate 
 * polynomials by evaluation and interpolation. The least N-1 variables are 
 * evaluated at sufficiently many points such that interpolation of their 
 * resultant can be performed. Essentially, this data structure stores the 
 * subresultant chains of the evaluated input polynomials.
 *
 * @w:         length of the subresultant chain
 * @dim:       if the input polynomials of N variables then N+1
 * @Sdims:     1-D array of length N+2
 *             Sdims[0] is not used
 *             Sdims[i] is the size of the i-th dimension of the SCube
 *             Sdims[N] = Sdims[N+1] = w  
 * @Ssize:     the size of the SCube as a 1-dim array 
 * @Sdata:     pointer to the actual data (= the multi-dim array)
 * @points_trees: pointer to the subproducttree data-structure
 * @SLcs:      the interpolated initials of the successive regular subresultants
 * @SPolys:    the interpolated regular subresultants
 * @doneBfr:   flag (see Maple code)
 *
 * @type:      the type of the chain, FFT or subproduct tree based chain
 * @layout:    the data layout of the chain, list of subresultants or transposed  
 * @Ses:       2^Ses[i] = Sdims[i] for i = 1 to N - 1, Ses[0] = 0
 *             Only valid if type = DftChain 
 * Return value: **/   
typedef struct ScubeST{
    sfixn w;  
    sfixn dim;
    sfixn *Sdims;
    sfixn Ssize;
    sfixn *Sdata;
    PTS_TREE *points_trees;
    preFFTRep **SLcs;
    preFFTRep **SPolys; 
    sfixn doneBfr;
} SCUBE_OLD;

/**
 * The subresultant data structure is organized into a 2-level layout. 
 * The outer layer is the common interface to other high level algorithms.
 * The inner layer can have different representations for the subresultant
 * chain data structure. 
 *
 * The supported (or to be supported) representations are:
 *
 * - subproduct tree based representation
 * - FFT based representation
 * - TFT based representation
 *
 * Each representation may have different layout. As of 09/20/2010, 
 * the subproduct tree representation and TFT uses a rectangle matrix 
 * to store data which doubles the space usage. This shortcoming 
 * should be fixed. We call this representation ListRectangle.
 *
 * There is another data layout arose from the GPU implementation, which
 * keep all the subresultants with the same index together. 
 *
 * For example, let 
 *
 * S02 = 0 + 1 * y + 2 * y^2    S12 = 6 +  7 * y + 8 * y^2
 * S01 = 3 + 4 * y              S11 = 9 + 10 * y 
 * S00 = 5                      S10 = 11
 *
 * The ListRectangle representation of [S2, S1, S0] is 
 *
 * 5   0   0    ## S00 
 * 3   4   0    ## S01
 * 0   1   2    ## S02
 * 
 * 11  0   0    ## S10
 * 9  10   0    ## S11
 * 6   7   8    ## S12
 *
 * The ListTriangle representation of [S2, S1, S0] is 
 *
 * 0   1   2    ## S02
 * 3   4        ## S01
 * 5            ## S00 
 * 
 * 6   7   8    ## S12
 * 9  10        ## S11
 * 11           ## S10
 *  
 * The StrideTriangle representation of [S2, S1, S0] is 
 *
 * 0   1   2     6   7   8  ## S02 and S12
 * 3   4         9  10      ## S01 and S11
 * 5            11          ## S00 and S10
 *
 * The reasons to have different representations is for 
 *
 * (1) the back-compatibility;
 * (2) conversion might be expensive. 
 * (3) extensible
 */

typedef enum { TreeChain,  // subproduct tree chain 
               DftChain,   // DFT chain
               TftChain,   // TFT chain
               CuDftChain  // GPU DFT chain
} ChainType_t;

// the same as SCUBE_OLD, only add the last field layout
typedef struct {
    sfixn w;                
    sfixn dim;
    sfixn *Sdims;
    sfixn Ssize;
    sfixn *Sdata;
    PTS_TREE *points_trees;
    preFFTRep **SLcs;
    preFFTRep **SPolys; 
    sfixn doneBfr;
} SPTreeChain_t; 

// cached coefficient polynomials are encoded as follows:
//
// For example, w = 4, number of coefficients is w * (w + 1) / 2 = 10.
//
// S00                 
// S10 S11             
// S20 S21 S22         
// S30 S31 S32 S33     
//
// The array of pointers is
//
// [CPtr0, CPtr1, CPtr2, CPtr3, CPtr4, CPtr5, CPtr6, CPtr7, CPtr8, CPtr9]
//
// The en(de)coding corresponding is 
//
// (i, d) <==> idx
//
// where idx = (1 + 2 + 3 + ... + i) + d
//
typedef struct {
    sfixn w;            // the size of the first subresultant
    sfixn dim;          // if the input polynomials of N variables then N + 1
    sfixn *Ses;         // the exponent of ith dimension size, length M + 1 = N
    sfixn *Sdims;       // 1-D array of length N + 2
                        // Sdims[0] is not used
                        // Sdims[i] is the size of the i-th dimension size
                        // Sdims[N] = Sdims[N+1] = w  
    sfixn *roots;       // primitive roots of unity
    sfixn Ssize;        // Sdata length
    sfixn *Sdata;       // actual data
    preFFTRep **Coeffs;   // the cached coefficients 
    preFFTRep **SPolys;   // the cached subresultants
} DftChain_t;

typedef struct {
    // no implemention yet
} TftChain_t;

// current only support bivariate or trivariate 
// could be extended by means of Kronecker substitutions
typedef struct {
    sfixn nvars;          // the number of variables of the input polynomials
    sfixn w;              // the size of the first subresultant
    sfixn *partial_sz_p;  // the partial size vector of first polynomial
    sfixn *partial_sz_q;  // the partial size vector of second polynomial
    void  *cudft_scube;   // the external cuda scube data structure
    preFFTRep **Coeffs;   // the cached coefficients 
    preFFTRep **SPolys;   // the cached subresultants
} CuDftChain_t;

typedef union {
    SPTreeChain_t *sptPtr;
    DftChain_t    *dftPtr;
    TftChain_t    *tftPtr;
    CuDftChain_t  *cudftPtr;
} ChainPtr_t;

typedef struct {
    ChainPtr_t  cPtr;
    ChainType_t cType; 
} SCUBE;

////////////////////
// Regular chains
////////////////////
typedef struct {
    preFFTRep *poly0;
    preFFTRep *poly1;
} regular_chain2;

typedef struct {
    preFFTRep *poly0;
    preFFTRep *poly1;
    preFFTRep *poly2;
} regular_chain3;

typedef union {
    regular_chain2 *pchain2;
    regular_chain3 *pchain3;
} regular_chain_ptr;

typedef enum {
    REG_CHAIN2,
    REG_CHAIN3
} regular_chain_type;

typedef struct {
    regular_chain_ptr chain;
    regular_chain_type type;
} regular_chain;

#endif
