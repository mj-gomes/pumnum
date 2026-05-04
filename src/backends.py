import astropy.units
import pint
import unyt

class BkAstropy:
    """Backend for astropy quantities"""
    def __init__(self, q):
        self.value = q.value
        self.units = q.unit

class BkPint:
    """Backend for pint quantities"""
    def __init__(self, q):
        self.value = q.magnitude
        self.units = q.units

class BkUnyt:
    """Backend for unyt quantities"""
    def __init__(self, q):
        self.value = q
        self.units = q.units
