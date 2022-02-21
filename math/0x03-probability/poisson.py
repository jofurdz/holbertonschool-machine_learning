#!/usr/bin/env python3
"""represents a poisson distribution"""


class Poisson():
    """class that represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """initializes class"""
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data)/len(data))

    def pmf(self, k):
        """returns PMF"""
        e = 2.7182818285
        pmf_numerator = (e ** (self.lambtha * -1) * (self.lambtha ** k))
        pmf_denominator = 1
        if type(k) is not int:
            k = int(k)
        for x in range(1, k + 1):
            pmf_denominator *= x
        return pmf_numerator / pmf_denominator

    def cdf(self, k):
        """returns CDF"""
        e = 2.7182818285
        cdf_store = []
        if type(k) is not int:
            k = int(k)
        for i in range(k + 1):
            cdf_numerator = (e ** (self.lambtha * -1) * (self.lambtha ** i))
            cdf_denominator = 1
            for x in range(1, i + 1):
                cdf_denominator *= x
            cdf_store.append(cdf_numerator / cdf_denominator)
        return sum(cdf_store)
