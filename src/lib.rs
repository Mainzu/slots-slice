pub type Slots<T> = [Option<T>];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    First,
    Last,
}

pub trait SlotsTrait<T>: AsRef<Slots<T>> {
    fn count(&self) -> usize {
        self.as_ref().iter().filter(|slot| slot.is_some()).count()
    }
    fn is_empty(&self) -> bool {
        self.count() == 0
    }
    fn is_full(&self) -> bool {
        self.count() == self.as_ref().len()
    }
    fn first_index(&self, occupied: bool) -> Option<usize> {
        self.as_ref()
            .iter()
            .position(move |slot| slot.is_some() == occupied)
    }
    fn first_value(&self) -> Option<&T> {
        self.as_ref().iter().find_map(|slot| slot.as_ref())
    }
    fn first_entry(&self) -> Option<(usize, &T)> {
        self.as_ref()
            .iter()
            .enumerate()
            .find_map(|(i, slot)| slot.as_ref().map(|v| (i, v)))
    }
    fn last_index(&self, occupied: bool) -> Option<usize> {
        self.as_ref()
            .iter()
            .rposition(move |slot| slot.is_some() == occupied)
    }

    fn last_value(&self) -> Option<&T> {
        self.as_ref().iter().find_map(|slot| slot.as_ref())
    }
    fn last_entry(&self) -> Option<(usize, &T)> {
        self.as_ref()
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, slot)| slot.as_ref().map(|v| (i, v)))
    }

    fn get_index(&self, direction: Direction, occupied: bool) -> Option<usize> {
        match direction {
            Direction::First => self.first_index(occupied),
            Direction::Last => self.last_index(occupied),
        }
    }
    fn get_value(&self, direction: Direction) -> Option<&T> {
        match direction {
            Direction::First => self.first_value(),
            Direction::Last => self.last_value(),
        }
    }
    fn get_entry(&self, direction: Direction) -> Option<(usize, &T)> {
        match direction {
            Direction::First => self.first_entry(),
            Direction::Last => self.last_entry(),
        }
    }
}
impl<T, A> SlotsTrait<T> for A where A: AsRef<Slots<T>> {}

pub trait SlotsMutTrait<T>: SlotsTrait<T> + AsMut<Slots<T>> {
    fn first_value_mut(&mut self) -> Option<&mut T> {
        self.as_mut().iter_mut().find_map(|slot| slot.as_mut())
    }
    fn first_entry_mut(&mut self) -> Option<(usize, &mut T)> {
        self.as_mut()
            .iter_mut()
            .enumerate()
            .find_map(|(i, slot)| slot.as_mut().map(|v| (i, v)))
    }
    fn last_value_mut(&mut self) -> Option<&mut T> {
        self.as_mut()
            .iter_mut()
            .rev()
            .find_map(|slot| slot.as_mut())
    }
    fn last_entry_mut(&mut self) -> Option<(usize, &mut T)> {
        self.as_mut()
            .iter_mut()
            .rev()
            .enumerate()
            .find_map(|(i, slot)| slot.as_mut().map(|v| (i, v)))
    }

    fn get_value_mut(&mut self, direction: Direction) -> Option<&mut T> {
        match direction {
            Direction::First => self.first_value_mut(),
            Direction::Last => self.last_value_mut(),
        }
    }
    fn get_entry_mut(&mut self, direction: Direction) -> Option<(usize, &mut T)> {
        match direction {
            Direction::First => self.first_entry_mut(),
            Direction::Last => self.last_entry_mut(),
        }
    }

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
    fn collapse(&mut self, direction: Direction) {
        match direction {
            Direction::First => self.collapse_front(),
            Direction::Last => self.collapse_back(),
        }
    }
    /// # Panic
    /// Panics if an index is out of bounds
    fn replace(&mut self, mapping: impl IntoIterator<Item = (usize, T)>) {
        let slots = self.as_mut();
        for (index, value) in mapping.into_iter() {
            slots[index] = Some(value);
        }
    }
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

pub fn indices<'a, T>(slots: &'a Slots<T>, occupied: bool) -> impl Iterator<Item = usize> + 'a {
    slots
        .iter()
        .enumerate()
        .filter_map(move |(i, slot)| (slot.is_some() == occupied).then_some(i))
}
pub fn values<'a, T>(slots: &'a Slots<T>) -> impl Iterator<Item = &'a T> + 'a {
    slots.iter().filter_map(|slot| slot.as_ref())
}
pub fn values_mut<'a, T>(slots: &'a mut Slots<T>) -> impl Iterator<Item = &'a mut T> + 'a {
    slots.iter_mut().filter_map(|slot| slot.as_mut())
}
pub fn entries<'a, T>(slots: &'a Slots<T>) -> impl Iterator<Item = (usize, &'a T)> + 'a {
    slots
        .iter()
        .enumerate()
        .filter_map(|(i, slot)| slot.as_ref().map(|v| (i, v)))
}
pub fn entries_mut<'a, T>(
    slots: &'a mut Slots<T>,
) -> impl Iterator<Item = (usize, &'a mut T)> + 'a {
    slots
        .iter_mut()
        .enumerate()
        .filter_map(|(i, slot)| slot.as_mut().map(|v| (i, v)))
}

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
#[macro_export]
macro_rules! array_of {
    ($max:expr, { $($index:expr => $value:expr,)* }) => {
        {
            let mut slots: [_; $max] = ::std::array::from_fn(|_| None);
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

        assert_eq!(slots.first_index(false), Some(0));
        assert_eq!(slots.first_index(true), Some(2));
        assert_eq!(slots.first_value(), Some(&'a'));
        assert_eq!(slots.first_entry(), Some((2, &'a')));

        assert_eq!(slots.last_index(false), Some(3));
        assert_eq!(slots.last_index(true), Some(4));
        assert_eq!(slots.last_value(), Some(&'b'));
        assert_eq!(slots.last_entry(), Some((4, &'b')));

        assert_eq!(
            slots.get_index(Direction::First, false),
            slots.first_index(false)
        );
        assert_eq!(
            slots.get_index(Direction::Last, true),
            slots.last_index(true)
        );
        assert_eq!(slots.get_value(Direction::First), slots.first_value());
        assert_eq!(slots.get_value(Direction::Last), slots.last_value());
        assert_eq!(slots.get_entry(Direction::First), slots.first_entry());
        assert_eq!(slots.get_entry(Direction::Last), slots.last_entry());
    }
    #[test]
    fn test_mut() {
        let mut slots = array_of!(6, {
            1 => 'a',
            3 => 'b',
            5 => 'C',
        });

        if let Some(value) = slots.first_value_mut() {
            value.make_ascii_uppercase();
        }
        assert_eq!(slots.first_value(), Some(&'A'));

        if let Some(value) = slots.last_value_mut() {
            value.make_ascii_lowercase();
        }
        assert_eq!(slots.last_value(), Some(&'c'));

        slots.collapse_front();
        assert_eq!(slots, [Some('A'), Some('b'), Some('c'), None, None, None]);

        slots.replace([(0, 'x'), (2, 'y'), (4, 'z')]);
        assert_eq!(
            slots,
            [Some('x'), Some('A'), Some('y'), Some('b'), Some('z'), None]
        );

        for (i, value) in entries_mut(&mut slots) {
            if i % 2 == 0 {
                value.make_ascii_uppercase();
            }
        }
        assert_eq!(
            slots,
            [Some('X'), Some('A'), Some('Y'), Some('b'), Some('Z'), None]
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
