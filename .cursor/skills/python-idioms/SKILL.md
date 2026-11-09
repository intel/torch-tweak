<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->
---
name: python-idioms
description: >-
  Idiomatic Python for Torch Tweak: aliases, tensor truth values, signatures,
  return shapes, protocols, generators, cleanup, and comments. Use when
  writing or editing Python under torch_tweak/, tests/, examples/, or tools/.
---

# Python idioms for Torch Tweak

- [Names refer to objects](#names-refer-to-objects)
- [Tensors](#tensors)
- [Signatures](#signatures)
- [Return one shape](#return-one-shape)
- [Classes stay small](#classes-stay-small)
- [Control flow](#control-flow)
- [Cleanup and import time](#cleanup-and-import-time)
- [Comments](#comments)
- [Cost](#cost)

Ruff already checks layout, imports, naming, docstrings, `print`, logging
f-strings, builtin shadowing, mutable defaults, and comprehension rewrites.
The enforced set is `tool.ruff.lint.select` in `pyproject.toml`.
Leave those to `uv run ruff check --fix`.

Line length is 120. Docstrings follow the Google convention Ruff checks.
This repo does not run a type checker in CI. Annotate a public signature
the way the surrounding module does. An annotation describes the value.
It does not change what the code does at run time.

Tests follow `.cursor/rules/unit-tests.mdc`.

Match the file you are editing. A pattern already used in that module wins.

## Names refer to objects

Assignment binds a name to an object. It does not copy that object.

A parameter is another name for the object the caller passed.
If the function appends to a list, writes a dict, or replaces a module
attribute, the caller sees that change.
Do this only when updating that object is the function's job.
`tune` updates the module in place.
Copy first when the function must keep a snapshot, and later writes must
leave the caller's object alone.

`+=` on a list changes that list. Every other name for the same list sees it.
`+=` on a number binds the local name to a new number.

A list or dict written on the class body is shared by every instance.
Create it in `__init__`, or use a dataclass `default_factory`.

Use `None` as a default only when `None` means "the caller left this out",
and `None` is not itself a valid input.
Build the fresh list or dict inside the function.
When `None` is a valid input, use a module-level sentinel and test it with `is`.

`is` asks whether two names refer to the same object. Use it for `None`
and sentinels.
`==` asks whether two values are equal.
CPython may reuse one object for a small int or a short string.
Do not treat that as a guarantee.

`list(items)`, `dict(mapping)`, and a list slice copy the container only.
The elements are still the original objects.
A tuple that holds lists can still have those lists mutated.

`[[]] * n` repeats one inner list. A comprehension builds a new one each time.

## Tensors

A tensor or a NumPy array is not a condition.
`if tensor`, `tensor and other`, and `if tensor == other` do not mean
what they mean for a Python number.
Test `.numel()`, the shape, or a reduction such as `.any()`.

Do not loop over tensor elements in Python when a tensor operator does
the same work. Do not call `list(tensor)` to look at the values.

A view uses the same storage as the tensor it came from.
Writing through the view changes the original.
Call `.clone()` when later code still needs the old values.
Inside code you are timing, clone only when the copy itself is what
you are measuring.

A tensor cannot be a dict key.
Cache on a hashable description: shape, dtype, device, and the flags
that change the compiled result.
After an object is used as a key, do not mutate it.

## Signatures

If a caller could mistake an option for something else in that position,
put the option after `*` so it has to be passed by name.
Use `*args` when the number of values really varies, as with a stack of
tensors.
A public function does not take `**kwargs`. A misspelled argument must raise.

Accept `Iterable` or `Sequence` when the function only reads the input.
Return a `list` or a `tuple` when the caller may loop over the result twice.

Skip a keyword argument when you would only repeat its default.
Pass it when this call is about that argument, or when you are forwarding
a variable.

A function you store or pass on is a `def`, so tracebacks show its name.
Use `lambda` only inside a larger expression.

A function defined inside a loop reads the loop variable when it is called,
which is after the loop has finished. It sees the last value.
Bind the value from this iteration:

```python
funcs.append(lambda i=i: i)
```

`functools.partial` does the same. Use `operator.itemgetter` or
`attrgetter` when the function only reads an item or an attribute.

## Return one shape

Each branch returns the same kind of value.
A value from one branch and `None` from another is one shape only when
the success value cannot itself be `None`.
If a stored `None` and a miss would look the same, raise the exception
callers already catch, or return a small named type with a field for
each fact.
A result with several fields is a dataclass or a `NamedTuple`, and every
field is set on every path.

A plain tuple is fine for two or three local values unpacked at the only
call site. A public function that hands back more than one value uses a
named type.

A generator is exhausted after one loop. Return one only when the series
is large and the caller loops once. A short series of modules, names, or
shapes is a list. Do not store a generator on an object and loop over it
later.

Pair two sequences with `zip`.
Prefer `first, *rest = items` to reading `items[0]` and `items[1]`.
Loop over a dict with `.items()`.

`.get` returns a value and does not insert a key.
A read uses `.get`. It does not assign a missing key.
`setdefault` inserts a missing key, and it always runs the default
expression, even when the key is already present. Keep that expression cheap.
Use `defaultdict` when every missing key should be created by a factory
that takes no arguments.

## Classes stay small

A group of fields is a dataclass, as backend configs and specs already are.
`slots=True` fits a small record created in large numbers.
Skip it when the class uses multiple inheritance.

A type that mainly does work is an ordinary class.
When two variants differ by one operation, pass a function.
Use a class with `__call__` when that operation must remember settings.
Do not add a base class and several subclasses for a single method.

Do not subclass `dict`, `list`, or `tuple` to add behavior.
C methods such as `dict.update` never call a Python `__setitem__` you wrote.
Hold the collection in a field of your own class.

A leading `_` means other modules should not use that name.
If they need the data, make it public: an attribute, a property, or a method.

A method that never uses `self` belongs on the module as a function.
Stored data is an attribute. A value that must be checked on write is a
property. Do not add `get_x` and `set_x`.

Ordinary code calls `len(obj)` and `iter(obj)`. The interpreter is what
calls `__len__` and `__iter__`.

If a mutable record defines `__eq__`, set `__hash__ = None`.
A frozen dataclass can be a dict key.
`__repr__` includes the type and the fields you need in a debugger.

A decorator that takes arguments is a function that returns the decorator.
Put `functools.wraps` on the inner function so the original name survives.
A single call site does not need a decorator.

## Control flow

The `else` on `for` or `while` runs when the loop did not `break`.
A `return` inside the body leaves the function first, so that `else`
does not run for that call either.
This search is already that form. Leave it in place:

```python
for key, value in pairs:
    if key == target:
        return value
else:
    return "missing"
```

Delete an `else` that always runs. It always runs when the body never
`break`s and never `return`s.
A `found = False` flag repeats the same search. Return from the loop,
or `break`, and use `else` for the miss.
The `else` on `try` runs when the body did not raise. Keep it when that
is the condition you need.

Catch a specific exception when the missing case is normal and the
attempt is cheap: a missing key, a missing attribute, a missing file.
Check first when the failure would be a long compile, a device error, or
a tensor used as a boolean.

Handle the specific exception before the broader one.
When you replace a caught error, write `raise NewError(...) from exc`.
The original stays attached. `raise NewError(...)` by itself drops it.
`from None` hides that original traceback. Use it only on purpose.
Do not catch `Exception` and keep going.

`match` fits a value with a known nested shape.
Use `if` when there are only two cases.
In a pattern, `case Color.RED` compares against that value.
`case red` stores the matched value under the name `red`.
Do not `match` on a tensor.

## Cleanup and import time

A `@contextmanager` does its setup, then `try`, `yield`, and `finally`.
The `finally` block releases the file, device, or graph, including when
the body raises.
Prefer that form unless the caller must inspect the object while it is open.
`__exit__` accepts the exception type, value, and traceback.
Do not free a resource in `__del__`.

Do not touch the device, download, compile, or take a lock while the
module is being imported. Do that work in the call that needs it.
A package `__init__` only re-exports.

Open text files with `encoding="utf-8"`. Use `pathlib` for paths.

## Comments

If you need a comment, take that as a warning: the code may not be clear
enough on its own.
Try a clearer name, a smaller function, or a named constant first.
Write a comment only for a fact the code cannot show.
Typical facts are a unit, a range, an encoding, aliasing, order, a device
quirk, or an invariant checked in another function. Name that function.

The comment describes the code that is there now.
It does not describe code you deleted, the old behavior, or why an
earlier version was different. History belongs in version control.
Do not add that note while changing or removing code.

Skip a comment that repeats the next line, walks a reviewer through
the change, or explains a term the reader of this file already knows.
One fact. Present tense. The same names the code uses.

## Cost

Time goes to compilation, moving data to the device, and Python loops
over tensor elements.
Looking up a handful of names can use a list.
A long lookup, or one you repeat, uses a set.

The device already runs its work in parallel. Wrapping a synchronous
public function in threads or `asyncio` does not speed that up.
Do not add a cache or `slots` on a path that is not hot.

Pass the message and its values to the logger, so a disabled level does
not build the string. Do not log an entire tensor.
