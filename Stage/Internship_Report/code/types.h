#ifndef _CUMODP_TYPES_H_
#define _CUMODP_TYPES_H_

/** portable integer types **/
#ifndef __modpn_integer_types__
#define __modpn_integer_types__

#include <ctype.h>
#define WINDOWS

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
typedef unsigned _int32  uint32;
#endif

#endif /* END OF INTEGER TYPES */

///////////////////////
// p = c * 2^npow + 1
///////////////////////
typedef struct {
    sfixn p;
    sfixn c;
    sfixn npow;
    sfixn rpow;    // 2^rpow is the smallest POT larger than p
    sfixn r;       // r = 2^rpow % p
    sfixn rnpow;   // rnpow = rpow - npow
} fprime_t;

////////////////////////////
//  Messages
////////////////////////////
typedef enum {
    CUMODP_SUCCESS                            = 0x000,
    CUMODP_FAILURE                            = 0x001,
    CUMODP_ASSUMPTION_ERROR                   = 0x002,
    // Device Query
    CUMODP_NO_CUDA_DEVICE                     = 0x100,
    CUMODP_HAS_CUDA_DEVICE                    = 0x101,
    // FFT
    CUMODP_FFT_ERROR                          = 0x200,
    CUMODP_FOURIER_DEGREE_TOO_SMALL           = 0x201,
    CUMODP_FFT_SIZE_TOO_LARGE                 = 0x202,
    CUMODP_FFT_SIZE_TOO_SMALL                 = 0x203,
    // Subresultant chain
    CUMODP_SUBRES_ERROR                       = 0x300,
    CUMODP_SUBRES_COARSE_ERROR                = 0x301,
    CUMODP_SUBRES_FINE_ERROR                  = 0x302,
    CUMODP_SUBRES_NON_REGULAR                 = 0x303,
    CUMODP_RES2_ERROR                         = 0x304,
    CUMODP_RES3_ERROR                         = 0x305,
    // Others
    CUMODP_OUT_OF_MEMORY                      = 0x401,
    CUMODP_UNKNOWN_ERROR                      = 0x999
} cumodp_err;

#endif // END_OF_FILE
