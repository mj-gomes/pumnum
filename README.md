**pumnum** (**P**hysical **U**nits **M**anipulation for **num**ba) is a package that allows to manipulate objects with physical units, using pint or unyt, while using numba jitted functions in nopython mode.

**Note: This package is in its very early stages of development and thus it is not yet stable.**

Current work-in-progress:
- [ ] add compatibility with astropy units
- [ ] make pumnum work for numba jitted nopython functions loops with product/division of physical quantities
- [ ] automatically convert units to SI (or other user defined) inside the decorator
