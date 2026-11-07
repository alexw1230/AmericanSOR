#define _XOPEN_SOURCE 700 /* for M_PI under -std=c11 */
#include "psor.h"
#include <math.h>
#include <string.h>

void setup_coef(const params_t *p, coef_t *c) {
    //PDE in log prices: u_tau = a u_xx + b u_x (+ same in y) - r u
    double a1 = 0.5 * p->sig1 * p->sig1;
    double a2 = 0.5 * p->sig2 * p->sig2;
    double b1 = p->r - a1;
    double b2 = p->r - a2;

    //Spacing puts global node N/2 exactly at (ln S10, ln S20)
    c->dt = p->T / p->M;
    c->hx = 2.0 * p->width * p->sig1 * sqrt(p->T) / p->N;
    c->hy = 2.0 * p->width * p->sig2 * sqrt(p->T) / p->N;
    c->x0 = log(p->S10) - (p->N / 2) * c->hx;
    c->y0 = log(p->S20) - (p->N / 2) * c->hy;

    double kx = c->dt * a1 / (c->hx * c->hx);
    double ky = c->dt * a2 / (c->hy * c->hy);
    double vx = c->dt * b1 / (2.0 * c->hx);
    double vy = c->dt * b2 / (2.0 * c->hy);

    //5 point stencil, same shape as the 3.4 matrix
    c->c0 = 1.0 + 2.0 * kx + 2.0 * ky + c->dt * p->r;
    c->cxp = kx + vx;
    c->cxm = kx - vx;
    c->cyp = ky + vy;
    c->cym = ky - vy;
}

double optimal_omega(const params_t *p, const coef_t *c) {
    //Jacobi spectral radius is exact for constant coefficients
    double rhoJ = 2.0 * (sqrt(c->cxp * c->cxm) + sqrt(c->cyp * c->cym))
                * cos(M_PI / (p->N + 1)) / c->c0;

    //Young's formula, red-black is consistently ordered
    return 2.0 / (1.0 + sqrt(1.0 - rhoJ * rhoJ));
}

//One half sweep over a single color, returns max |change| on this rank
static double psor_sweep(const params_t *p, const coef_t *c, local_t *L,
                         double omega, int color) {
    int W = L->W;
    double *u = L->u;
    const double *rhs = L->rhs;
    const double *g = L->g;
    double inv_c0 = 1.0 / c->c0;
    double maxdiff = 0.0;

    for (int i = 1; i <= L->nx; i++) {
        //Color uses global indices so every rank agrees on the checkerboard
        int gi = L->offx + i;
        int jstart = 1 + ((gi + L->offy + 1 + color) & 1);

        for (int j = jstart; j <= L->ny; j += 2) {
            size_t k = (size_t)i * W + j;
            double old = u[k];

            //Gauss-Seidel value, then over relax
            double gs = (rhs[k]
                         + c->cxp * u[k + W] + c->cxm * u[k - W]
                         + c->cyp * u[k + 1] + c->cym * u[k - 1]) * inv_c0;
            double val = old + omega * (gs - old);

            //Projection, never worth less than exercising now
            if (!p->euro && val < g[k]) {
                val = g[k];
            }

            double d = fabs(val - old);
            if (d > maxdiff) {
                maxdiff = d;
            }
            u[k] = val;
        }
    }
    return maxdiff;
}

result_t psor_solve(const params_t *p, const coef_t *c, local_t *L, double omega) {
    result_t res;
    size_t n = (size_t)(L->nx + 2) * L->W;
    double t_comm = 0.0;
    long iters = 0;

    //Value at expiry is the payoff
    memcpy(L->u, L->g, n * sizeof(double));

    MPI_Barrier(L->cart);
    double t_start = MPI_Wtime();

    for (int step = 1; step <= p->M; step++) {
        double tau = step * c->dt;

        set_boundary(p, c, L, tau);
        memcpy(L->rhs, L->u, n * sizeof(double));

        //Warm start from the last time step
        for (int it = 0; it < p->maxit; it++) {
            double t0;

            double d_red = psor_sweep(p, c, L, omega, 0);
            t0 = MPI_Wtime();
            exchange_halos(L);
            t_comm += MPI_Wtime() - t0;

            double d_black = psor_sweep(p, c, L, omega, 1);
            t0 = MPI_Wtime();
            exchange_halos(L);
            t_comm += MPI_Wtime() - t0;

            double d_loc = (d_red > d_black) ? d_red : d_black;
            double d_all;
            t0 = MPI_Wtime();
            MPI_Allreduce(&d_loc, &d_all, 1, MPI_DOUBLE, MPI_MAX, L->cart);
            t_comm += MPI_Wtime() - t0;

            iters++;
            if (d_all < p->tol) {
                break;
            }
        }
    }

    res.t_local = MPI_Wtime() - t_start;
    MPI_Allreduce(&res.t_local, &res.t_total, 1, MPI_DOUBLE, MPI_MAX, L->cart);

    res.iters = iters;
    res.t_comm = t_comm;
    res.price = center_price(p, L);
    res.hash = grid_hash(p, L);
    return res;
}
