#ifndef PARAMS_H
#define PARAMS_H

//Model, grid and run options
typedef struct {
    //Market and contract
    double S10, S20;   //Spot prices
    double K;          //Strike
    double r;          //Risk free rate
    double sig1, sig2; //Volatilities (uncorrelated)
    double T;          //Maturity in years
    double width;      //Domain half width in std devs

    //Grid and solver
    int N;             //Interior nodes per dimension
    int M;             //Time steps
    double omega;      //Relaxation factor, 0 = theoretical best
    double tol;        //Stop when max |change| < tol
    int maxit;         //Max PSOR iterations per time step

    //Run options
    int euro;          //1 = European, no projection
    int reps;
    const char *csv;
} params_t;

//Grid spacing and implicit Euler stencil (constant in log space)
typedef struct {
    double dt, hx, hy;
    double x0, y0;     //Log price at global node 0
    double c0;         //Diagonal
    double cxp, cxm;   //Neighbors i+1, i-1
    double cyp, cym;   //Neighbors j+1, j-1
} coef_t;

#endif
