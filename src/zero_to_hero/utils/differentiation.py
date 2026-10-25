def derivative_numerical(f, pt, h: float = 0.001):
    # symmetric approximation
    return (f(pt+h) - f(pt-h)) / (2*h)