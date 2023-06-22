# Slots Slice

[![Latest Version](https://img.shields.io/crates/v/slots-slice)](https://crates.io/crates/slots-slice)

This is a small crate that aims provides utilities for manipulating slices of optional values, referred to as [`Slots<T>`](crate::Slots).

## Features

- Conveniently manipulate slices of optional values.
- Perform operations on slots such as counting, checking emptiness and fullness, and retrieving values and entries.
- Macro syntax for creating and assigning.

## Usage

Bring the prelude into scope:

```rust
use slots_slice::prelude::*;
```

The highlight of the crate are [`SlotsTrait`](crate::SlotsTrait) and [`SlotsMutTrait`](crate::SlotsMutTrait) which add methods for accessing and manipulating slots immutably and mutably. These operate on anything that implements [`AsRef`](core::convert::AsRef)<[`[T]`](https://doc.rust-lang.org/std/primitive.slice.html)> so they are available right away on structs such as array and [`Vec<T>`](std::vec::Vec).

Overview of [`SlotsTrait`](crate::SlotsTrait):

```rust
use slots_slice::prelude::*;

let slots = [None, Some('a'), None, Some('b')];

assert_eq!(slots.count(), 2);
assert!(!slots.is_empty());
assert!(!slots.is_full());

assert_eq!(slots.front_index(true), Some(1));
assert_eq!(slots.front_value(), Some(&'a'));
assert_eq!(slots.front_entry(), Some((1, &'a')));
```

[`SlotsMutTrait`](crate::SlotsMutTrait) provide the mutable version of `SlotsTrait` as well as collapse functionality.

```rust
use slots_slice::prelude::*;

let mut slots = [None, Some('a'), None, Some('b')];

assert_eq!(slots.front_value_mut(), Some(&mut 'a'));
assert_eq!(slots.front_entry_mut(), Some((1, &mut 'a')));

slots.collapse_front();

assert_eq!(slots, [Some('a'), Some('b'), None, None]);
```
