//! Iterate over all permutations of fixed length from some set.
//!
//! Code courtesy of <https://rosettacode.org/wiki/Permutations_with_repetitions#Rust>
pub(crate) struct PermutationIterator<'a, T: 'a> {
    universe: &'a [T],
    size: usize,
    prev: Option<Vec<usize>>,
}

pub(crate) fn permutations<T>(universe: &[T], size: usize) -> PermutationIterator<T> {
    PermutationIterator {
        universe,
        size,
        prev: None,
    }
}

fn map<T>(values: &[T], ixs: &[usize]) -> Vec<T>
where
    T: Clone,
{
    ixs.iter().map(|&i| values[i].clone()).collect()
}

impl<'a, T> Iterator for PermutationIterator<'a, T>
where
    T: Clone,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        let n = self.universe.len();

        if n == 0 {
            return None;
        }

        match self.prev {
            None => {
                let zeroes: Vec<usize> = std::iter::repeat(0).take(self.size).collect();
                let result = Some(map(self.universe, &zeroes[..]));
                self.prev = Some(zeroes);
                result
            }
            Some(ref mut indexes) => match indexes.iter().position(|&i| i + 1 < n) {
                None => None,
                Some(position) => {
                    for index in indexes.iter_mut().take(position) {
                        *index = 0;
                    }
                    indexes[position] += 1;
                    Some(map(self.universe, &indexes[..]))
                }
            },
        }
    }
}
