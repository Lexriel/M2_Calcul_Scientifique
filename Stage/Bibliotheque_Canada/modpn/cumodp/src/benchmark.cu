#include <iostream>
#include <ctime>
#include "../include/cumodp.h"
#include "../include/rdr_poly.h"
#include "../include/scube.h"
#include "../include/fft_aux.h"
#include "../include/list_stockham.h"
#include "../include/stockham.h"

using std::cout;
using std::endl;

void rdr_poly_tst() {
    sfixn n = 3;
    sfixn ns[] = {2, 3, 3};
    sfixn coeffs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 16, 17};
    rdr_poly F(n, ns, coeffs);
    F.print_to_maple();   
}

void fftmul_uni_tst(int argc, char** argv) {
    sfixn p = 469762049;
    sfixn d1 = 4325376, d2 = d1;
    if (argc > 2) {
        d1 = atoi(argv[1]);
        d2 = atoi(argv[2]);
    }
    check_polymul(d1, d2, p);
}

struct random_vector {
    explicit random_vector(size_t len, sfixn p = 257) : length(len) {
        data = new sfixn[length];
        for (size_t i = 0; i < length; ++i) {
            data[i] = rand() % p;
        }
    }
 
    // A randon vector such as a polynomial it has x - a factor.
    // The generation is as follows:
    // (1) generate a random vector [c_1, c_2, ..., c_{n-1}] 
    // (2) then set
    //      b_0 = - c_1 * a
    //      b_1 = c_1 - c_2 * a
    //      b_2 = c_2 - c_3 * a
    //
    //      ...
    //
    //      b_{n - 2} = c_{n - 2} - c_{n - 1} * a
    //      b_{n - 1} = c_{n - 1}
    //
    // This can be done in-place.
    //
    random_vector(size_t n, sfixn p, sfixn a) : length(n) {
        data = new sfixn[n];
        for (size_t i = 1; i <= n - 1; ++i) {
            data[i] = rand() % p;
        }
        
        data[0] = neg_mod(mul_mod(data[1], a, p), p); 
        for (size_t i = 1; i <= n - 2; ++i) {
            data[i] = sub_mod(data[i], mul_mod(data[i+1], a, p), p);    
        }
    }
 
    // random_vector with last m element given by A   
    random_vector(size_t n, sfixn p, size_t m, const sfixn *A) : length(n) 
    {
        assert(n >= m);
        data = new sfixn[n];
        for (size_t i = 0; i < n - m; ++i) {
            data[i] = rand() % p;
        }
        for (size_t i = 0; i < m; ++i) {
            data[i + n - m] = A[i] % p;
        }
    }

    ~random_vector() { delete [] data; }
 
    void print() {
        size_t line_words = 16;
        for (size_t i = 0; i < length; ++i) {
            if (i % line_words == 0) printf("\n##");
            printf("%4d ", data[i]);
        }
        printf("\n");
    }
        
    sfixn *data;
    size_t length;
};

void subres2_tst(int argc, char** argv) {
    sfixn fp = 469762049;
    //sfixn fp = 257;
    //sfixn fp = 13313;
    sfixn npx = 110, npy = npx; 
    sfixn nqx = 110, nqy = nqx; 
    if (argc >= 5) {
        npx = atoi(argv[1]);
        npy = atoi(argv[2]);
        nqx = atoi(argv[3]);
        nqy = atoi(argv[4]);
    }

    //srand(time(NULL));
    sfixn *initP = new sfixn[npx]();
    sfixn *initQ = new sfixn[nqx]();
    initP[0] = fp - 1;
    initQ[0] = fp - 1;
    initP[npx - 1] = 1;
    initQ[nqx - 1] = 1;

    random_vector PV(npx * npy, fp, npx, initP); 
    random_vector QV(nqx * nqy, fp, nqx, initQ); 
    sfixn *P = PV.data;
    sfixn *Q = QV.data;

    //PV.print();
    //QV.print();
    delete [] initP;
    delete [] initQ;

    scube_t scb(npx, npy, nqx, nqy, fp);
    // scb.info();
    bool success = scb.build_scube_data2(npx, P, nqx, Q);
    if (!success) { printf("FFT based method failed\n"); return; }
    // scb.info();

    int u = 3, v = 1; 
    scube_t::subres_coeff_type res = scb.subres_coeff(u, v);
    //scb.info();
    //scb.subres_coeff(0, 0);
    //scb.subres_coeff(0, 0);
    //scb.subres_coeff(2, 0);
    //scb.subres_coeff(2, 0);
    //scb.info();
    //printf("p := %d;\n", fp);
    //printf("npx := %d;\n", npx);
    //printf("npy := %d;\n", npy);
    //printf("nqx := %d;\n", nqx);
    //printf("nqy := %d;\n", nqy);
    //printf("u := %d;\n", u);
    //printf("v := %d;\n", v);
    //printf("A := [");
    //for (int i = 0; i < npx * npy - 1; ++i)
    //    printf("%d,", P[i]);
    //printf("%d]:\n", P[npx * npy - 1]);

    //printf("B := [");
    //for (int i = 0; i < nqx * nqy - 1; ++i)
    //    printf("%d,", Q[i]);
    //printf("%d]:\n", Q[nqx * nqy - 1]);

    //printf("E := [");
    //for (int i = 0; i < scb.coefficient_size() - 1; ++i)
    //    printf("%d,", res[i]);
    //printf("%d]:\n", res[scb.coefficient_size() - 1]);
}

void subres3_tst(int argc, char** argv) {
    sfixn fp = 469762049;
    //sfixn fp = 13313;
    sfixn npx = 5, npy = npx, npz = npx; 
    sfixn nqx = 5, nqy = nqx, nqz = nqx; 
    if (argc >= 7) {
        npx = atoi(argv[1]);
        npy = atoi(argv[2]);
        npz = atoi(argv[3]);
        nqx = atoi(argv[4]);
        nqy = atoi(argv[5]);
        nqz = atoi(argv[6]);
    }

    //srand(time(NULL));
    sfixn *initP = new sfixn[npx*npy]();
    sfixn *initQ = new sfixn[nqx*nqy]();
    initP[0] = initQ[0] = fp - 1;
    initP[npx * npy - 1] = 1;
    initQ[nqx * nqy - 1] = 1;

    random_vector PV(npx * npy * npz, fp, npx * npy, initP); 
    random_vector QV(nqx * nqy * nqz, fp, nqx * nqy, initQ); 
    sfixn *P = PV.data;
    sfixn *Q = QV.data;

    //PV.print();
    //QV.print();
    delete [] initP;
    delete [] initQ;
    
    scube_t scb(npx, npy, npz, nqx, nqy, nqz, fp);
    scb.info();
    bool success = scb.build_scube_data3(npx, npy, P, nqx, nqy, Q);
    if (!success) { printf("FFT based method failed\n"); return; }

    int u = 0, v = 0; 
    scube_t::subres_coeff_type res = scb.subres_coeff(u, v);
    scb.info();

    //printf("p := %d;\n", fp);
    //printf("npx := %d;\n", npx);
    //printf("npy := %d;\n", npy);
    //printf("npz := %d;\n", npz);
    //printf("nqx := %d;\n", nqx);
    //printf("nqy := %d;\n", nqy);
    //printf("nqz := %d;\n", nqz);
    //printf("u := %d;\n", u);
    //printf("v := %d;\n", v);
    //printf("Bx := %d;\n", 1<<scb.get_bounds_exp(0));
    //printf("A := [");
    //for (int i = 0; i < npx * npy * npz - 1; ++i)
    //    printf("%d,", P[i]);
    //printf("%d]:\n", P[npx * npy * npz- 1]);

    //printf("B := [");
    //for (int i = 0; i < nqx * nqy * npz - 1; ++i)
    //    printf("%d,", Q[i]);
    //printf("%d]:\n", Q[nqx * nqy * npz - 1]);

    //printf("E := [");
    //for (int i = 0; i < scb.coefficient_size() - 1; ++i)
    //    printf("%d,", res[i]);
    //printf("%d]:\n", res[scb.coefficient_size() - 1]);
}

int main(int argc, char** argv) {
    subres2_tst(argc, argv);
    return 0;
}
