"""One registration mechanism for every pluggable kind of thing — engines and device probes.

The same idea as Omnia's ``ProviderRegistry``: a class declares itself with a decorator, the
registry maps a NAME to the class, and configuration refers to it by that name. Adding an engine
or a device backend is therefore one subclass and one decorator — nothing else learns about it.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → class, for one kind of plug-in (``kind`` is only used in error messages)."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Class decorator: make ``cls`` available as ``name``.

        Raises:
            ValueError: when ``name`` is already taken by a different class — two plug-ins
                silently answering to one name is how a config ends up running the wrong one.
        """

        def decorate(cls: type[T]) -> type[T]:
            existing = self._items.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(f"{self._kind} {name!r} is already registered by {existing!r}")
            self._items[name] = cls
            cls.registry_name = name  # type: ignore[attr-defined]
            return cls

        return decorate

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"unknown {self._kind} {name!r}; known: {', '.join(self.names()) or 'none'}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> list[tuple[str, type[T]]]:
        return sorted(self._items.items())
