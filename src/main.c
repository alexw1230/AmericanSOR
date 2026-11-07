#include "params.h"
#include "black_scholes.h"
#include "grid.h"
#include "psor.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CSV_HEADER "P,px,py,N,M,type,omega,rep,iters,price,euro_exact,abs_err," \
                   "hash,t_total,t_per_iter,t_comm_avg,t_comp_max,t_comp_avg"

static int parse_args(int argc, char **argv, params_t *p) {
    //Test parameters
    p->S10 = 100.0;
    p->S20 = 100.0;
    p->K = 100.0;
    p->r = 0.05;
    p->sig1 = 0.30;
    p->sig2 = 0.20;
    p->T = 1.0;
    p->width = 5.0;

    //Defaults
    p->N = 0;
    p->M = 100;
    p->omega = 0.0;
    p->tol = 1e-8;
    p->maxit = 100000;
    p->euro = 0;
    p->reps = 1;
    p->csv = "results.csv";

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has_val = (i + 1 < argc);
        if (!strcmp(a, "-n") && has_val) {
            p->N = atoi(argv[++i]);
        } else if (!strcmp(a, "-m") && has_val) {
            p->M = atoi(argv[++i]);
        } else if (!strcmp(a, "-r") && has_val) {
            p->reps = atoi(argv[++i]);
        } else if (!strcmp(a, "-w") && has_val) {
            p->omega = atof(argv[++i]);
        } else if (!strcmp(a, "-t") && has_val) {
            p->tol = atof(argv[++i]);
        } else if (!strcmp(a, "-o") && has_val) {
            p->csv = argv[++i];
        } else if (!strcmp(a, "-e")) {
            p->euro = 1;
        } else {
            return -1;
        }
    }
    if (p->N < 2 || p->M < 1 || p->reps < 1) {
        return -1;
    }
    return 0;
}

static FILE *open_csv(const char *path, const char *header) {
    FILE *f = fopen(path, "a");
    if (!f) {
        perror(path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f, "%s\n", header);
    }
    return f;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    params_t p;
    if (parse_args(argc, argv, &p) != 0) {
        if (world_rank == 0) {
            fprintf(stderr, "bad arguments (-n N is required); see README for options\n");
        }
        MPI_Finalize();
        return 1;
    }

    coef_t c;
    local_t L;
    setup_coef(&p, &c);
    grid_setup(&p, &c, &L);

    double omega = (p.omega > 0.0) ? p.omega : optimal_omega(&p, &c);
    double exact = euro_geo_put(sqrt(p.S10 * p.S20), p.T, &p);

    //Central differences need these >= 0 for diagonal dominance
    if (L.rank == 0 && (c.cxm < 0.0 || c.cym < 0.0)) {
        fprintf(stderr, "warning: grid too coarse, not diagonally dominant\n");
    }

    FILE *f = NULL;
    if (L.rank == 0) {
        f = open_csv(p.csv, CSV_HEADER);
    }

    for (int rep = 0; rep < p.reps; rep++) {
        result_t res = psor_solve(&p, &c, &L, omega);

        //Comm vs comp split, imbalance = max / avg compute
        double t_comp = res.t_local - res.t_comm;
        double comm_sum, comp_sum, comp_max;
        MPI_Reduce(&res.t_comm, &comm_sum, 1, MPI_DOUBLE, MPI_SUM, 0, L.cart);
        MPI_Reduce(&t_comp, &comp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, L.cart);
        MPI_Reduce(&t_comp, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, L.cart);

        if (L.rank == 0) {
            const char *type = p.euro ? "european" : "american";
            printf("N=%d M=%d P=%d (%dx%d) %s rep=%d omega=%.4f iters=%ld "
                   "price=%.8f T_total=%.4e comm=%.1f%% hash=%016llx\n",
                   p.N, p.M, L.P, L.dims[0], L.dims[1], type, rep, omega,
                   res.iters, res.price, res.t_total,
                   100.0 * (comm_sum / L.P) / res.t_total,
                   (unsigned long long)res.hash);
            fflush(stdout);

            if (f) {
                fprintf(f, "%d,%d,%d,%d,%d,%s,%.6f,%d,%ld,%.12f,%.12f,%.3e,"
                           "%016llx,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                        L.P, L.dims[0], L.dims[1], p.N, p.M, type, omega, rep,
                        res.iters, res.price, exact, fabs(res.price - exact),
                        (unsigned long long)res.hash, res.t_total,
                        res.t_total / res.iters, comm_sum / L.P,
                        comp_max, comp_sum / L.P);
                fflush(f);
            }
        }
    }

    if (f) {
        fclose(f);
    }
    grid_free(&L);
    MPI_Finalize();
    return 0;
}
