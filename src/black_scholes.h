#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#include "params.h"

//Closed form European put on G = sqrt(S1*S2) with tau left to expiry
double euro_geo_put(double G, double tau, const params_t *p);

//Put payoff on the geometric average at log prices (x, y)
double geo_payoff(double x, double y, double K);

#endif
