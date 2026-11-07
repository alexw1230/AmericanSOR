#ifndef GRID_H
#define GRID_H

#include <mpi.h>
#include <stdint.h>
#include "params.h"

//This rank's block of the 2D grid
typedef struct {
    MPI_Comm cart;
    int rank, P;
    int dims[2];       //px, py
    int coords[2];
    int xlo, xhi;      //Neighbor ranks, MPI_PROC_NULL at the edge
    int ylo, yhi;
    int nx, ny;        //Owned interior nodes
    int offx, offy;    //Global index of local node 0 (ghost)
    int W;             //Row stride = ny + 2
    MPI_Datatype col_t;
    double *u;         //Solution with ghosts, (nx+2) x (ny+2)
    double *rhs;       //Previous time level
    double *g;         //Payoff at each node
} local_t;

//Cartesian process grid, block split, buffers and payoff
void grid_setup(const params_t *p, const coef_t *c, local_t *L);

//Free buffers and MPI types
void grid_free(local_t *L);

//Dirichlet values on the physical boundary at time to expiry tau
void set_boundary(const params_t *p, const coef_t *c, local_t *L, double tau);

//Swap one ghost layer with the 4 neighbors
void exchange_halos(local_t *L);

//Order independent hash of the interior, same on every P iff grids match
uint64_t grid_hash(const params_t *p, const local_t *L);

//Value at the center node (S10, S20)
double center_price(const params_t *p, const local_t *L);

#endif
