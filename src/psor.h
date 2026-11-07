#ifndef PSOR_H
#define PSOR_H

#include <stdint.h>
#include "params.h"
#include "grid.h"

typedef struct {
    long iters;        //PSOR iterations over all time steps
    double price;
    uint64_t hash;
    double t_total;    //Max over ranks
    double t_local;    //This rank
    double t_comm;     //This rank, halos + allreduce
} result_t;

//Grid spacing and implicit Euler stencil from the model parameters
void setup_coef(const params_t *p, coef_t *c);

//Theoretical best omega for the linear (European) problem
double optimal_omega(const params_t *p, const coef_t *c);

//Full solve, marches backward from expiry to today
result_t psor_solve(const params_t *p, const coef_t *c, local_t *L, double omega);

#endif
