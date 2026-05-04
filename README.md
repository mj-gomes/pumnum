**pumnum** (**P**hysical **U**nits **M**anipulation for **num**ba) is a package that allows to manipulate objects with physical units, using pint or unyt, while using numba jitted functions in nopython mode.

**Note: This package is in its very early stages of development and thus it is not yet stable.**

Current TO-DO list:
- [ ] make pumnum work for numba jitted nopython functions loops with product/division of physical quantities https://github.com/mj-gomes/pumnum/issues/1
- [ ] add compatibility with astropy units https://github.com/mj-gomes/pumnum/issues/2
- [ ] automatically convert units to SI (or other user defined) inside the decorator https://github.com/mj-gomes/pumnum/issues/3
