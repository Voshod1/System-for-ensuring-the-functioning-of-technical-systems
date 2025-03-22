# System-for-ensuring-the-functioning-of-technical-systems

It's an implementation of a computational algorithm for diagnosing the technical condition system. The work considers a resuscitation vehicle that moves in operating mode, i.e. with a patient on board. The patient's life is ensured by medical equipment powered by the onboard network of the resuscitation vehicle.

Data descryption:

Y1 - On-board network voltage.
X11 - Measured battery voltage.
X12 - Crankshaft rotation speed.
X13 - Power provided by the auxiliary generator.
X14 - Total power consumption.

Y2 - Amount of fuel left.
X21 - Crankshaft rotation speed.
X22 - Power provided by the auxiliary generator.

Y3 - Battery voltage.
X31 - Crankshaft rotation speed.
X32 - Power provided by the auxiliary generator.
X33 - Total power consumption.


Chebyshev polynomials were chosen as the basic approximating functions.
