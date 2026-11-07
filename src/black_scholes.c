#include "black_scholes.h"
#include <math.h>

static double norm_cdf(double z) {
    return 0.5 * erfc(-z / sqrt(2.0));
}

double euro_geo_put(double G, double tau, const params_t *p) {
    if (tau <= 0.0) {
        return (p->K > G) ? p->K - G : 0.0;
    }

    //sqrt(S1*S2) is lognormal, acts like one asset with vol sG and yield q
    double s2 = p->sig1 * p->sig1 + p->sig2 * p->sig2;
    double sG = 0.5 * sqrt(s2);
    double q = s2 / 8.0;

    //Regular Black-Scholes put from here
    double sq = sG * sqrt(tau);
    double d1 = (log(G / p->K) + (p->r - q + 0.5 * sG * sG) * tau) / sq;
    double d2 = d1 - sq;

    return p->K * exp(-p->r * tau) * norm_cdf(-d2)
         - G * exp(-q * tau) * norm_cdf(-d1);
}

double geo_payoff(double x, double y, double K) {
    double v = K - exp(0.5 * (x + y));
    return (v > 0.0) ? v : 0.0;
}
