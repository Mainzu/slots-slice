#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::invalid_rust_codeblocks)]
#![deny(rustdoc::missing_crate_level_docs)]
#![warn(rustdoc::invalid_codeblock_attributes)]
pub type Slots<T> = [Option<T>];

pub mod prelude {
    pub use crate as slots_slice;
    pub use crate::{array_of, replace, Direction, Slots, SlotsMutTrait, SlotsTrait};
}

/// Represents the direction for searching slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Search from the first slot to the last slot.
    Front,
    /// Search from the last slot to the first slot.
    Back,
}

/// Provides utility methods for accessing and manipulating a collection of slots.
///
/// Content: counting, accessing indices, values, and entries.
pub trait SlotsTrait<T>: AsRef<Slots<T>> {
    /// Returns the number of occupied slots in the collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsTrait;
    ///
    /// let slots = [None, Some('a'), Some('b'), None];
    /// assert_eq!(slots.count(), 2);
    /// ```
    fn count(&self) -> usize {
        self.as_ref().iter().filter(|slot| slot.is_some()).count()
    }

    /// Checks if the collection is empty, i.e., all slots are empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsTrait;
    ///
    /// let slots: [Option<()>; 3] = [None, None, None];
    /// assert!(slots.is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Checks if the collection is full, i.e., all slots are occupied.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsTrait;
    ///
    /// let slots = [Some('a'), Some('b'), Some('c')];
    /// assert!(slots.is_full());
    /// ```
    fn is_full(&self) -> bool {
        self.count() == self.as_ref().len()
    }

    /// Returns the index of the first slot that matches the occupancy state.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::{SlotsTrait, Direction};
    ///
    /// let slots = [None, Some('a'), None, Some('b')];
    /// assert_eq!(slots.front_index(false), Some(0));
    /// assert_eq!(slots.front_index(true), Some(1));
    /// ```
    fn front_index(&self, occupied: bool) -> Option<usize> {
        self.as_ref()
            .iter()
            .position(move |slot| slot.is_some() == occupied)
    }

    /// Returns a reference to the value of the first occupied slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsTrait;
    ///
    /// let slots = [None, Some('a'), Some('b')];
    /// assert_eq!(slots.front_value(), Some(&'a'));
    /// ```
    fn front_value(&self) -> Option<&T> {
        self.as_ref().iter().find_map(|slot| slot.as_ref())
    }

    /// Returns the index and a reference to the value of the first occupied slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsTrait;
    ///
    /// let slots = [None, Some('a'), Some('b')];
    /// assert_eq!(slots.front_entry(), Some((1, &'a')));
    /// ```
    fn front_entry(&self) -> Option<(usize, &T)> {
        self.as_ref()
            .iter()
            .enumerate()
            .find_map(|(i, slot)| slot.as_ref().map(|v| (i, v)))
    }

    /// Returns the index of the last slot that matches the occupancy state.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::{SlotsTrait, Direction};
    ///
    /// let slots = [None, Some('a'), None, Some('b')];
    /// assert_eq!(slots.back_index(false), Some(2));
    /// assert_eq!(slots.back_index(true), Some(3));
    /// ```
    fn back_index(&self, occupied: bool) -> Option<usize> {
        self.as_ref()
            .iter()
            .rposition(move |slot| slot.is_some() == occupied)
    }

    /// Returns a reference to the value of the last occupied slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsTrait;
    ///
    /// let slots = [Some('a'), Some('b'), None];
    /// assert_eq!(slots.back_value(), Some(&'b'));
    /// ```
    fn back_value(&self) -> Option<&T> {
        self.as_ref().iter().rev().find_map(|slot| slot.as_ref())
    }

    /// Returns the index and a reference to the value of the last occupied slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsTrait;
    ///
    /// let slots = [Some('a'), Some('b'), None];
    /// assert_eq!(slots.back_entry(), Some((1, &'b')));
    /// ```
    fn back_entry(&self) -> Option<(usize, &T)> {
        self.as_ref()
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, slot)| slot.as_ref().map(|v| (i, v)))
    }

    /// Returns the index of the slot in the specified direction that matches the occupancy state.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::{SlotsTrait, Direction};
    ///
    /// let slots = [None, Some('a'), None, Some('b')];
    /// assert_eq!(slots.get_index(Direction::Front, false), slots.front_index(false));
    /// assert_eq!(slots.get_index(Direction::Front, true), slots.front_index(true));
    /// assert_eq!(slots.get_index(Direction::Back, false), slots.back_index(false));
    /// assert_eq!(slots.get_index(Direction::Back, true), slots.back_index(true));
    /// ```
    fn get_index(&self, direction: Direction, occupied: bool) -> Option<usize> {
        match direction {
            Direction::Front => self.front_index(occupied),
            Direction::Back => self.back_index(occupied),
        }
    }

    /// Returns a reference to the value of the slot in the specified direction.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::{SlotsTrait, Direction};
    ///
    /// let slots = [Some('a'), None, Some('b')];
    /// assert_eq!(slots.get_value(Direction::Front), slots.front_value());
    /// assert_eq!(slots.get_value(Direction::Back), slots.back_value());
    /// ```
    fn get_value(&self, direction: Direction) -> Option<&T> {
        match direction {
            Direction::Front => self.front_value(),
            Direction::Back => self.back_value(),
        }
    }

    /// Returns the index and a reference to the value of the slot in the specified direction.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::{SlotsTrait, Direction};
    ///
    /// let slots = [Some('a'), None, Some('b')];
    /// assert_eq!(slots.get_entry(Direction::Front), slots.front_entry());
    /// assert_eq!(slots.get_entry(Direction::Back), slots.back_entry());
    /// ```
    fn get_entry(&self, direction: Direction) -> Option<(usize, &T)> {
        match direction {
            Direction::Front => self.front_entry(),
            Direction::Back => self.back_entry(),
        }
    }
}

impl<T, A> SlotsTrait<T> for A where A: AsRef<Slots<T>> {}

/// Provides mutable access and manipulation methods for a collection of slots.
///
/// This trait extends the [`SlotsTrait<T>`] trait and adds additional methods for mutating the slots.
/// It is implemented for types that can be converted to a mutable reference of [`Slots<T>`].
pub trait SlotsMutTrait<T>: SlotsTrait<T> + AsMut<Slots<T>> {
    /// Returns a mutable reference to the value of the first occupied slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsMutTrait;
    ///
    /// let mut slots = [Some('a'), None, Some('b')];
    /// if let Some(value) = slots.front_value_mut() {
    ///     *value = 'c';
    /// }
    /// assert_eq!(slots[0], Some('c'));
    /// ```
    fn front_value_mut(&mut self) -> Option<&mut T> {
        self.as_mut().iter_mut().find_map(|slot| slot.as_mut())
    }

    /// Returns the index and a mutable reference to the value of the first occupied slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsMutTrait;
    ///
    /// let mut slots = [Some('a'), None, Some('b')];
    /// if let Some((index, value)) = slots.front_entry_mut() {
    ///     *value = 'c';
    /// }
    /// assert_eq!(slots[0], Some('c'));
    /// ```
    fn front_entry_mut(&mut self) -> Option<(usize, &mut T)> {
        self.as_mut()
            .iter_mut()
            .enumerate()
            .find_map(|(i, slot)| slot.as_mut().map(|v| (i, v)))
    }

    /// Returns a mutable reference to the value of the last occupied slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsMutTrait;
    ///
    /// let mut slots = [Some('a'), None, Some('b')];
    /// if let Some(value) = slots.back_value_mut() {
    ///     *value = 'c';
    /// }
    /// assert_eq!(slots[2], Some('c'));
    /// ```
    fn back_value_mut(&mut self) -> Option<&mut T> {
        self.as_mut()
            .iter_mut()
            .rev()
            .find_map(|slot| slot.as_mut())
    }

    /// Returns the index and a mutable reference to the value of the last occupied slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsMutTrait;
    ///
    /// let mut slots = [Some('a'), None, Some('b')];
    /// if let Some((index, value)) = slots.back_entry_mut() {
    ///     *value = 'c';
    /// }
    /// assert_eq!(slots[2], Some('c'));
    /// ```
    fn back_entry_mut(&mut self) -> Option<(usize, &mut T)> {
        self.as_mut()
            .iter_mut()
            .rev()
            .enumerate()
            .find_map(|(i, slot)| slot.as_mut().map(|v| (i, v)))
    }

    /// Returns a mutable reference to the value of the slot in the specified direction.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::{SlotsMutTrait, Direction};
    ///
    /// let mut slots = [Some('a'), None, Some('b')];
    /// if let Some(value) = slots.get_value_mut(Direction::Front) {
    ///     *value = 'c';
    /// }
    /// assert_eq!(slots[0], Some('c'));
    ///
    /// if let Some(value) = slots.get_value_mut(Direction::Back) {
    ///     *value = 'd';
    /// }
    /// assert_eq!(slots[2], Some('d'));
    /// ```
    fn get_value_mut(&mut self, direction: Direction) -> Option<&mut T> {
        match direction {
            Direction::Front => self.front_value_mut(),
            Direction::Back => self.back_value_mut(),
        }
    }

    /// Returns the index and a mutable reference to the value of the slot in the specified direction.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::{SlotsMutTrait, Direction};
    ///
    /// let mut slots = [Some('a'), None, Some('b')];
    /// if let Some((index, value)) = slots.get_entry_mut(Direction::Front) {
    ///     *value = 'c';
    /// }
    /// assert_eq!(slots[0], Some('c'));
    ///
    /// if let Some((index, value)) = slots.get_entry_mut(Direction::Back) {
    ///     *value = 'd';
    /// }
    /// assert_eq!(slots[2], Some('d'));
    /// ```
    fn get_entry_mut(&mut self, direction: Direction) -> Option<(usize, &mut T)> {
        match direction {
            Direction::Front => self.front_entry_mut(),
            Direction::Back => self.back_entry_mut(),
        }
    }

    /// Collapses the slots towards the front by removing unoccupied slots and shifting occupied slots to the left.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsMutTrait;
    ///
    /// let mut slots = [None, Some('a'), None, Some('b')];
    /// slots.collapse_front();
    /// assert_eq!(slots, [Some('a'), Some('b'), None, None]);
    /// ```
    fn collapse_front(&mut self) {
        let slots = self.as_mut();
        let mut write_index = 0;

        for read_index in 0..slots.len() {
            if slots[read_index].is_some() {
                slots[write_index] = slots[read_index].take();
                write_index += 1;
            }
        }
    }

    /// Collapses the slots towards the back by removing unoccupied slots and shifting occupied slots to the right.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsMutTrait;
    ///
    /// let mut slots = [None, Some('a'), None, Some('b')];
    /// slots.collapse_back();
    /// assert_eq!(slots, [None, None, Some('a'), Some('b')]);
    /// ```
    fn collapse_back(&mut self) {
        let slots = self.as_mut();
        let mut write_index = slots.len();

        for read_index in (0..slots.len()).rev() {
            if slots[read_index].is_some() {
                write_index -= 1;
                slots[write_index] = slots[read_index].take();
            }
        }
    }

    /// Collapses the slots in the specified direction by removing unoccupied slots and shifting occupied slots accordingly.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::{SlotsMutTrait, Direction};
    ///
    /// let mut slots = [None, Some('a'), None, Some('b')];
    /// slots.collapse(Direction::Front);
    /// assert_eq!(slots, [Some('a'), Some('b'), None, None]);
    ///
    /// let mut slots = [None, Some('a'), None, Some('b')];
    /// slots.collapse(Direction::Back);
    /// assert_eq!(slots, [None, None, Some('a'), Some('b')]);
    /// ```
    fn collapse(&mut self, direction: Direction) {
        match direction {
            Direction::Front => self.collapse_front(),
            Direction::Back => self.collapse_back(),
        }
    }

    /// Replaces the values of the slots based on the provided index-value pairs.
    ///
    /// The method takes an iterator of `(index, value)` pairs and replaces the values of the slots at the specified indices with the provided values.
    ///
    /// # Panics
    ///
    /// This method panics if an index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsMutTrait;
    ///
    /// let mut slots = [None, None, None];
    /// slots.replace([(0, 'a'), (2, 'b')]);
    /// assert_eq!(slots, [Some('a'), None, Some('b')]);
    /// ```
    fn replace(&mut self, mapping: impl IntoIterator<Item = (usize, T)>) {
        let slots = self.as_mut();
        for (index, value) in mapping.into_iter() {
            slots[index] = Some(value);
        }
    }

    /// Tries to replace the values of the slots based on the provided index-value pairs, returning an error if any index is out of bounds.
    ///
    /// The method takes an iterator of `(index, value)` pairs and tries to replace the values of the slots at the specified indices with the provided values.
    /// If any index is out of bounds, it returns an `Err` variant with the `(index, value)` pair that caused the error and the remaining iterator.
    /// If all replacements are successful, it returns an `Ok` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use slots_slice::SlotsMutTrait;
    ///
    /// let mut slots = [None, None, None];
    ///
    /// let result = slots.try_replace([(0, 'a'), (2, 'b')]);
    /// assert!(result.is_ok());
    /// assert_eq!(slots, [Some('a'), None, Some('b')]);
    ///
    /// let result = slots.try_replace([(1, 'c'), (3, 'd')]);
    /// assert!(result.is_err());
    /// let ((index, value), remaining) = result.unwrap_err();
    /// assert_eq!(index, 3);
    /// assert_eq!(value, 'd');
    /// assert_eq!(remaining.collect::<Vec<_>>(), vec![]);
    ///
    /// ```
    fn try_replace<M: IntoIterator<Item = (usize, T)>>(
        &mut self,
        mapping: M,
    ) -> Result<(), ((usize, T), M::IntoIter)> {
        let slots = self.as_mut();
        let mut iter = mapping.into_iter();
        while let Some((index, value)) = iter.next() {
            if let Some(slot) = slots.get_mut(index) {
                *slot = Some(value);
            } else {
                return Err(((index, value), iter));
            }
        }
        Ok(())
    }
}
impl<T, A> SlotsMutTrait<T> for A where A: SlotsTrait<T> + AsMut<Slots<T>> {}

/// Returns an iterator over the indices of occupied or unoccupied slots in the `slots`.
///
/// # Example
///
/// ```
/// use slots_slice;
///
/// let slots = [None, None, Some('a'), None, Some('b')];
/// let occupied_indices: Vec<usize> = slots_slice::indices(&slots, true).collect();
/// assert_eq!(occupied_indices, vec![2, 4]);
///
/// let unoccupied_indices: Vec<usize> = slots_slice::indices(&slots, false).collect();
/// assert_eq!(unoccupied_indices, vec![0, 1, 3]);
/// ```
pub fn indices<'a, T>(slots: &'a Slots<T>, occupied: bool) -> impl Iterator<Item = usize> + 'a {
    slots
        .iter()
        .enumerate()
        .filter_map(move |(i, slot)| (slot.is_some() == occupied).then_some(i))
}

/// Returns an iterator over the values of occupied slots in the `slots`.
///
/// # Example
///
/// ```
/// use slots_slice;
///
/// let slots = [None, None, Some('a'), None, Some('b')];
/// let occupied_values: Vec<&char> = slots_slice::values(&slots).collect();
/// assert_eq!(occupied_values, vec![&'a', &'b']);
/// ```
pub fn values<'a, T>(slots: &'a Slots<T>) -> impl Iterator<Item = &'a T> + 'a {
    slots.iter().filter_map(|slot| slot.as_ref())
}

/// Returns an iterator over mutable references to the values of occupied slots in the `slots`.
///
/// # Example
///
/// ```
/// use slots_slice;
///
/// let mut slots = [None, None, Some('a'), None, Some('b')];
/// for value in slots_slice::values_mut(&mut slots) {
///     *value = 'x';
/// }
/// assert_eq!(slots, [None, None, Some('x'), None, Some('x')]);
/// ```
pub fn values_mut<'a, T>(slots: &'a mut Slots<T>) -> impl Iterator<Item = &'a mut T> + 'a {
    slots.iter_mut().filter_map(|slot| slot.as_mut())
}

/// Returns an iterator over the entries (index, value) of occupied slots in the `slots`.
///
/// # Example
///
/// ```
/// use slots_slice;
///
/// let slots = [None, None, Some('a'), None, Some('b')];
/// let occupied_entries: Vec<(usize, &char)> = slots_slice::entries(&slots).collect();
/// assert_eq!(occupied_entries, vec![(2, &'a'), (4, &'b')]);
/// ```
pub fn entries<'a, T>(slots: &'a Slots<T>) -> impl Iterator<Item = (usize, &'a T)> + 'a {
    slots
        .iter()
        .enumerate()
        .filter_map(|(i, slot)| slot.as_ref().map(|v| (i, v)))
}

/// Returns an iterator over mutable references to the entries (index, value) of occupied slots in the `slots`.
///
/// # Example
///
/// ```
/// use slots_slice;
///
/// let mut slots = [None, None, Some('a'), None, Some('b')];
/// for (index, value) in slots_slice::entries_mut(&mut slots) {
///     *value = 'x';
/// }
/// assert_eq!(slots, [None, None, Some('x'), None, Some('x')]);
/// ```
pub fn entries_mut<'a, T>(
    slots: &'a mut Slots<T>,
) -> impl Iterator<Item = (usize, &'a mut T)> + 'a {
    slots
        .iter_mut()
        .enumerate()
        .filter_map(|(i, slot)| slot.as_mut().map(|v| (i, v)))
}

/// Replaces the values at specified indices in the `slots` with the provided values.
///
/// This macro allows replacing multiple values in the `slots` at once. It takes a set of index-value pairs enclosed in curly braces.
///
/// # Examples
///
/// ```
/// use slots_slice::replace;
///
/// let mut slots: [Option<u32>; 5] = [None; 5];
///
/// slots_slice::replace!(slots, { 1 => 42, 3 => 99 });
///
/// assert_eq!(slots, [None, Some(42), None, Some(99), None]);
/// ```
///
/// # Panics
/// As this macro uses [`ops::Index`](core::ops::Index), it may panic if any index is out of bounds.
#[macro_export]
macro_rules! replace {
    ($slots:expr, { $($index:expr => $value:expr,)* }) => {
        {
            let slots = $slots.as_mut();
            $(
                slots[$index] = Some($value);
            )*
        }
    };
    ($slots:expr, { $($index:expr => $value:expr),* }) => {
        replace!($slots, { $($index => $value,)* })
    };
}

/// Creates an array of [`Option<T>`] with specified values at corresponding indices.
///
/// This macro allows creating an array of `Option<T>` with provided values at specified indices.
/// The array size is determined by the first argument specified. Useful for large slots.
///
/// # Examples
///
/// ```
/// use slots_slice::array_of;
///
/// let slots: [Option<char>; 5] = slots_slice::array_of!(5, { 1 => 'a', 3 => 'b' });
///
/// assert_eq!(slots, [None, Some('a'), None, Some('b'), None]);
/// ```
#[macro_export]
macro_rules! array_of {
    ($max:expr, { $($index:expr => $value:expr,)* }) => {
        {
            let mut slots: [_; $max] = ::core::array::from_fn(|_| None);
            $(
                slots[$index] = Some($value);
            )*
            slots
        }
    };
    ($max:expr, { $($index:expr => $value:expr),* }) => {
        array_of!($max, { $($index => $value,)* })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn create() {
        assert_eq!(
            [None, None, Some('a'), None, Some('b')],
            array_of!(5, {
                2 => 'a',
                4 => 'b',
            })
        );
    }

    #[test]
    fn test() {
        let slots = array_of!(5, {
            2 => 'a',
            4 => 'b',
        });

        assert_eq!(slots, [None, None, Some('a'), None, Some('b')]);

        assert_eq!(slots.count(), 2);
        assert_eq!(slots.is_empty(), false);
        assert_eq!(slots.is_full(), false);

        assert_eq!(slots.front_index(false), Some(0));
        assert_eq!(slots.front_index(true), Some(2));
        assert_eq!(slots.front_value(), Some(&'a'));
        assert_eq!(slots.front_entry(), Some((2, &'a')));

        assert_eq!(slots.back_index(false), Some(3));
        assert_eq!(slots.back_index(true), Some(4));
        assert_eq!(slots.back_value(), Some(&'b'));
        assert_eq!(slots.back_entry(), Some((4, &'b')));

        assert_eq!(
            slots.get_index(Direction::Front, false),
            slots.front_index(false)
        );
        assert_eq!(
            slots.get_index(Direction::Back, true),
            slots.back_index(true)
        );
        assert_eq!(slots.get_value(Direction::Front), slots.front_value());
        assert_eq!(slots.get_value(Direction::Back), slots.back_value());
        assert_eq!(slots.get_entry(Direction::Front), slots.front_entry());
        assert_eq!(slots.get_entry(Direction::Back), slots.back_entry());
    }
    #[test]
    fn test_mut() {
        let mut slots = array_of!(6, {
            1 => 'a',
            3 => 'b',
            5 => 'C',
        });

        if let Some(value) = slots.front_value_mut() {
            value.make_ascii_uppercase();
        }
        assert_eq!(slots.front_value(), Some(&'A'));

        if let Some(value) = slots.back_value_mut() {
            value.make_ascii_lowercase();
        }
        assert_eq!(slots.back_value(), Some(&'c'));

        slots.collapse_front();
        assert_eq!(slots, [Some('A'), Some('b'), Some('c'), None, None, None]);

        slots.replace([(0, 'x'), (2, 'y'), (4, 'z')]);
        assert_eq!(
            slots,
            [Some('x'), Some('b'), Some('y'), None, Some('z'), None]
        );

        for (i, value) in entries_mut(&mut slots) {
            if i % 2 == 0 {
                value.make_ascii_uppercase();
            }
        }
        assert_eq!(
            slots,
            [Some('X'), Some('b'), Some('Y'), None, Some('Z'), None]
        );
    }

    #[test]
    fn c_front() {
        let mut slots = [None, Some('a'), None, Some('b'), None, Some('c')];
        slots.collapse_front();
        assert_eq!(slots, [Some('a'), Some('b'), Some('c'), None, None, None]);
    }
    #[test]
    fn c_back() {
        let mut slots = [None, Some('a'), None, Some('b'), None, Some('c')];
        slots.collapse_back();
        assert_eq!(slots, [None, None, None, Some('a'), Some('b'), Some('c')]);
    }
    #[test]
    fn replace() {
        let mut slots = [None, None, None, None, None, None];
        slots.replace([(1, 'a'), (2, 'b'), (3, 'c')]);
        assert_eq!(slots, [None, Some('a'), Some('b'), Some('c'), None, None]);
    }
    #[test]
    fn try_replace_ok() {
        let mut slots = [None, None, None, None, None, None];
        assert_eq!(
            slots
                .try_replace([(1, 'a'), (2, 'b'), (3, 'c')])
                .map_err(|(entry, it)| (entry, it.collect::<Vec<_>>())),
            Ok(())
        );
        assert_eq!(slots, [None, Some('a'), Some('b'), Some('c'), None, None]);
    }
    #[test]
    fn try_replace_err() {
        let mut slots = [None, None, None, None, None, None];
        assert_eq!(
            slots
                .try_replace([(1, 'a'), (6, 'f'), (2, 'b'), (3, 'c')])
                .map_err(|(entry, it)| (entry, it.collect::<Vec<_>>())),
            Err(((6, 'f'), vec![(2, 'b'), (3, 'c')]))
        );
    }
    #[test]
    fn replace_macro() {
        let mut slots = [None, None, None, None, None, None];
        replace!(slots, {
            1 => 'a',
            2 => 'b',
            3 => 'c',
        });
        assert_eq!(slots, [None, Some('a'), Some('b'), Some('c'), None, None]);
    }
    #[test]
    fn slots_arr() {
        let slots = array_of!(7, {
            0 => 'a',
            2 => 'b',
            5 => 'c',
        });
        assert_eq!(
            slots,
            [Some('a'), None, Some('b'), None, None, Some('c'), None]
        );
    }
    #[test]
    fn iter_occupied_indices() {
        let slots = array_of!(6, {
            1 => 'a',
            3 => 'b',
            5 => 'c',
        });
        let mut indices = indices(&slots, true);
        assert_eq!(indices.next(), Some(1));
        assert_eq!(indices.next(), Some(3));
        assert_eq!(indices.next(), Some(5));
        assert_eq!(indices.next(), None);
    }
    #[test]
    fn iter_empty_indices() {
        let slots = array_of!(6, {
            1 => 'a',
            3 => 'b',
            5 => 'c',
        });
        let mut indices = indices(&slots, false);
        assert_eq!(indices.next(), Some(0));
        assert_eq!(indices.next(), Some(2));
        assert_eq!(indices.next(), Some(4));
        assert_eq!(indices.next(), None);
    }
    #[test]
    fn iter_values() {
        let slots = array_of!(6, {
            1 => 'a',
            3 => 'b',
            5 => 'c',
        });
        let mut values = values(&slots);
        assert_eq!(values.next(), Some(&'a'));
        assert_eq!(values.next(), Some(&'b'));
        assert_eq!(values.next(), Some(&'c'));
        assert_eq!(values.next(), None);
    }
    #[test]
    fn iter_values_mut() {
        let mut slots = array_of!(6, {
            1 => 'a',
            3 => 'b',
            5 => 'c',
        });
        for value in values_mut(&mut slots) {
            *value = 'x';
        }
        assert_eq!(slots, array_of!(6, { 1 => 'x', 3 => 'x', 5 => 'x' }));
    }
    #[test]
    fn iter_entries() {
        let slots = array_of!(6, {
            1 => 'a',
            3 => 'b',
            5 => 'c',
        });
        let mut entries = entries(&slots);
        assert_eq!(entries.next(), Some((1, &'a')));
        assert_eq!(entries.next(), Some((3, &'b')));
        assert_eq!(entries.next(), Some((5, &'c')));
        assert_eq!(entries.next(), None);
    }
    #[test]
    fn iter_entries_mut() {
        let mut slots = array_of!(6, {
            1 => 'a',
            3 => 'b',
            5 => 'c',
        });
        for (i, value) in entries_mut(&mut slots) {
            if i == 3 {
                *value = 'x';
            }
        }
        assert_eq!(slots, array_of!(6, { 1 => 'a', 3 => 'x', 5 => 'c' }));
    }
}
