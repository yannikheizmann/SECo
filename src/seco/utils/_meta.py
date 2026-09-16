from abc import ABC, ABCMeta


class RegistryMeta(ABCMeta):
    """
    Metaclass for automatic registration of classes into a registry system.

    When a class using this metaclass is defined, it automatically creates an instance,
    retrieves its `_name` attribute, and registers the instance into all registries
    declared on its base classes (via the `__registry__` attribute).

    This facilitates the registry pattern by reducing boilerplate code for registration.

    Attributes:
        None (metaclass behavior operates at class creation time).

    Raises:
        TypeError: If registration fails or `_name` attribute is missing.
    """

    def __init__(cls, name, bases, clsdict):
        super().__init__(name, bases, clsdict)

        if getattr(cls, "__skip_registry__", False):
            return

        registries = set()
        for base in cls.__mro__[1:]:
            registry = getattr(base, "__registry__", None)
            if registry:
                registries.add(registry)

        if not registries:
            return

        if not ABC in cls.__bases__:
            try:
                instance = cls()
                name = getattr(instance, "_name", None)
                if not name:
                    raise ValueError(f"{cls.__name__}._name is missing")
                for registry in registries:
                    registry.register(name, instance)
            except Exception as e:
                raise TypeError(f"Failed to register '{cls.__name__}': {e}")