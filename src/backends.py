import astropy.units
import pint
import unyt

class Backend():
    def convert_to_si():
        return

class BkAstropy(Backend):
    """Backend for astropy quantities"""
    def __init__(self, q):
        self.quantity = q
        self.magnitude = q.value
        self.units = q.unit

    def convert_to_unit_system(self, system):
        avail_sys = ["si", "cgs", "astrophys", "imperial", "misc", "photometric", "required_by_vounit"]
        if system in avail_sys:
            return getattr(self.quantity, system)
        else:
            raise ValueError(f"Unknown system: {system!r}. Available: {', '.join(sorted(avail_sys))}")

class BkPint(Backend):
    """Backend for pint quantities"""
    def __init__(self, q):
        self.quantity = q
        self.magnitude = q.magnitude
        self.units = q.units

    def convert_to_unit_system(self, system):
        ureg = pint.UnitRegistry()
        avail_sys = dir(ureg.sys)
        if system in avail_sys:
            # in order to perform the conversion, we need to create a new UnitRegistry
            # with the selected unit system, and a new pint Quantity associated with it.
            ureg_conversion = pint.UnitRegistry(system=system)
            self.quantity = ureg_conversion.Quantity(self.quantity.magnitude, str(self.quantity.units))
            return self.quantity.to_base_units()
        else:
            raise ValueError(f"Unknown system: {system!r}. Available: {', '.join(sorted(avail_sys))}")

class BkUnyt(Backend):
    """Backend for unyt quantities"""
    def __init__(self, q):
        self.quantity = q
        self.magnitude = q.value
        self.units = q.units

    def convert_to_unit_system(self, system):
        avail_sys = list(unyt.unit_systems.unit_system_registry)
        if system in avail_sys:
            self.quantity.convert_to_base(system)
            return self.quantity
        else:
            raise ValueError(f"Unknown system: {system!r}. Available: {', '.join(sorted(avail_sys))}")
