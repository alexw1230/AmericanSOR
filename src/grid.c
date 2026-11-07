#include "grid.h"
#include "black_scholes.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Split n nodes over np ranks as evenly as possible
static void split_1d(int n, int np, int coord, int *count, int *offset) {
    int base = n / np;
    int rem = n % np;
    *count = base + (coord < rem ? 1 : 0);
    *offset = coord * base + (coord < rem ? coord : rem);
}

static double *must_alloc(size_t count) {
    double *p = malloc(count * sizeof(double));
    if (!p) {
        fprintf(stderr, "allocation of %zu doubles failed\n", count);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return p;
}

void grid_setup(const params_t *p, const coef_t *c, local_t *L) {
    int periods[2] = {0, 0};

    MPI_Comm_size(MPI_COMM_WORLD, &L->P);
    L->dims[0] = 0;
    L->dims[1] = 0;
    MPI_Dims_create(L->P, 2, L->dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, L->dims, periods, 1, &L->cart);
    MPI_Comm_rank(L->cart, &L->rank);
    MPI_Cart_coords(L->cart, L->rank, 2, L->coords);
    MPI_Cart_shift(L->cart, 0, 1, &L->xlo, &L->xhi);
    MPI_Cart_shift(L->cart, 1, 1, &L->ylo, &L->yhi);

    split_1d(p->N, L->dims[0], L->coords[0], &L->nx, &L->offx);
    split_1d(p->N, L->dims[1], L->coords[1], &L->ny, &L->offy);
    L->W = L->ny + 2;

    //Column of nx values, one row apart
    MPI_Type_vector(L->nx, 1, L->W, MPI_DOUBLE, &L->col_t);
    MPI_Type_commit(&L->col_t);

    size_t n = (size_t)(L->nx + 2) * L->W;
    L->u = must_alloc(n);
    L->rhs = must_alloc(n);
    L->g = must_alloc(n);

    for (int i = 0; i <= L->nx + 1; i++) {
        double x = c->x0 + (L->offx + i) * c->hx;
        for (int j = 0; j <= L->ny + 1; j++) {
            double y = c->y0 + (L->offy + j) * c->hy;
            L->g[(size_t)i * L->W + j] = geo_payoff(x, y, p->K);
        }
    }
}

void grid_free(local_t *L) {
    MPI_Type_free(&L->col_t);
    MPI_Comm_free(&L->cart);
    free(L->u);
    free(L->rhs);
    free(L->g);
}

//European: closed form. American: max(closed form, payoff)
static double boundary_value(const params_t *p, const coef_t *c,
                             int gi, int gj, double tau) {
    double x = c->x0 + gi * c->hx;
    double y = c->y0 + gj * c->hy;
    double v = euro_geo_put(exp(0.5 * (x + y)), tau, p);

    if (!p->euro) {
        double pay = geo_payoff(x, y, p->K);
        if (pay > v) {
            v = pay;
        }
    }
    return v;
}

void set_boundary(const params_t *p, const coef_t *c, local_t *L, double tau) {
    int W = L->W;

    //Only ranks on the edge of the process grid own boundary ghosts
    if (L->coords[0] == 0) {
        for (int j = 1; j <= L->ny; j++) {
            L->u[j] = boundary_value(p, c, 0, L->offy + j, tau);
        }
    }
    if (L->coords[0] == L->dims[0] - 1) {
        size_t row = (size_t)(L->nx + 1) * W;
        for (int j = 1; j <= L->ny; j++) {
            L->u[row + j] = boundary_value(p, c, p->N + 1, L->offy + j, tau);
        }
    }
    if (L->coords[1] == 0) {
        for (int i = 1; i <= L->nx; i++) {
            L->u[(size_t)i * W] = boundary_value(p, c, L->offx + i, 0, tau);
        }
    }
    if (L->coords[1] == L->dims[1] - 1) {
        for (int i = 1; i <= L->nx; i++) {
            L->u[(size_t)i * W + L->ny + 1] =
                boundary_value(p, c, L->offx + i, p->N + 1, tau);
        }
    }
}

void exchange_halos(local_t *L) {
    int W = L->W;
    int nx = L->nx;
    int ny = L->ny;
    double *u = L->u;

    //Rows are contiguous
    MPI_Sendrecv(&u[nx * W + 1], ny, MPI_DOUBLE, L->xhi, 0,
                 &u[1], ny, MPI_DOUBLE, L->xlo, 0,
                 L->cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&u[W + 1], ny, MPI_DOUBLE, L->xlo, 1,
                 &u[(nx + 1) * W + 1], ny, MPI_DOUBLE, L->xhi, 1,
                 L->cart, MPI_STATUS_IGNORE);

    //Columns use the strided type
    MPI_Sendrecv(&u[W + ny], 1, L->col_t, L->yhi, 2,
                 &u[W], 1, L->col_t, L->ylo, 2,
                 L->cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&u[W + 1], 1, L->col_t, L->ylo, 3,
                 &u[W + ny + 1], 1, L->col_t, L->yhi, 3,
                 L->cart, MPI_STATUS_IGNORE);
}

//splitmix64 finalizer
static uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint64_t grid_hash(const params_t *p, const local_t *L) {
    uint64_t h = 0;

    //Hash each value with its global index, XOR doesn't care about order
    for (int i = 1; i <= L->nx; i++) {
        for (int j = 1; j <= L->ny; j++) {
            uint64_t bits;
            memcpy(&bits, &L->u[(size_t)i * L->W + j], sizeof(bits));
            uint64_t gidx = (uint64_t)(L->offx + i) * (uint64_t)(p->N + 2)
                          + (uint64_t)(L->offy + j);
            h ^= mix64(bits ^ mix64(gidx));
        }
    }

    uint64_t hall = 0;
    MPI_Allreduce(&h, &hall, 1, MPI_UINT64_T, MPI_BXOR, L->cart);
    return hall;
}

double center_price(const params_t *p, const local_t *L) {
    int gc = p->N / 2;
    double v = -DBL_MAX;

    if (gc > L->offx && gc <= L->offx + L->nx &&
        gc > L->offy && gc <= L->offy + L->ny) {
        v = L->u[(size_t)(gc - L->offx) * L->W + (gc - L->offy)];
    }

    double vall;
    MPI_Allreduce(&v, &vall, 1, MPI_DOUBLE, MPI_MAX, L->cart);
    return vall;
}
