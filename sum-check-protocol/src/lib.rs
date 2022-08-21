use std::marker::PhantomData;

use ark_ff::{Field, Zero};
use ark_poly::{
    multivariate::{self, SparseTerm, Term},
    polynomial::DenseMVPolynomial,
    univariate, Polynomial,
};
use ark_std::rand::Rng;
use bitvec::slice::BitSlice;

/// An error type of sum check protocol
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("prover claim mismatches evaluation {0} {1}")]
    ProverClaimMismatch(String, String),
}

/// A convenient way to iterate over $n$-dimentional boolean hypercube.
pub struct BooleanHypercube<F: Field> {
    n: u32,
    current: u64,
    __f: PhantomData<F>,
}

impl<F: Field> BooleanHypercube<F> {
    /// Create an $n$-dimentional [`BooleanHypercube`]
    pub fn new(n: u32) -> Self {
        Self {
            n,
            current: 0,
            __f: PhantomData,
        }
    }
}

impl<F: Field> Iterator for BooleanHypercube<F> {
    type Item = Vec<F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == 2u64.pow(self.n) {
            None
        } else {
            let vec = self.current.to_le_bytes();
            let s: &BitSlice<u8> = BitSlice::try_from_slice(&vec).unwrap();
            self.current += 1;

            Some(
                s.iter()
                    .take(self.n as usize)
                    .map(|f| match *f {
                        false => F::zero(),
                        true => F::one(),
                    })
                    .collect(),
            )
        }
    }
}

/// The state of the Prover.
pub struct Prover<F: Field, P: SumCheckPolynomial<F>> {
    /// $g$ a polynomial being used in this run of the protocol.
    g: P,

    /// $C_1$ a value prover _claims_ equal the true answer.
    c_1: F,

    /// Random values $r_1,...,r_j$ sent by the [`Verifier`] in the
    /// previous rounds.
    r: Vec<F>,
}

impl<F: Field, P: SumCheckPolynomial<F>> Prover<F, P> {
    /// Create a new [`Prover`] state with the polynomial $g$.
    pub fn new(g: P) -> Self {
        let c_1 = g.to_evaluations().into_iter().sum();
        let num_vars = g.num_vars();
        Self {
            g,
            c_1,
            r: Vec::with_capacity(num_vars),
        }
    }

    /// Get the value $C_1$ that prover claims equal true answer.
    pub fn c_1(&self) -> F {
        self.c_1
    }

    /// Perform $j$-th round of the [`Prover`] side of the prococol.
    pub fn round(&mut self, r_prev: F, j: usize) -> Option<univariate::SparsePolynomial<F>> {
        if j != 0 {
            self.r.push(r_prev);
        }
        multivariate_to_univariate_with_fixed_vars(&self.g, &self.r, j)
    }
}

/// An abstraction over all types of polynomials that may be
/// used in a sumcheck protocol.
pub trait SumCheckPolynomial<F: Field> {
    /// Evaluates `self` at a given point
    ///
    /// Return `None` if dimentionality of `point` does not match
    /// an expected one.
    fn evaluate(&self, point: &[F]) -> Option<F>;

    /// Given an index of a variable `i`, and a point `at` in
    /// $F^n$ reduce the multivariate polynomial
    /// to a single variate polynomial $s(x_i)$ by partially
    /// evaluating g at point $(a_1,...,a_{i-1},X_i,a_{i+1},...,a_n)$.
    ///
    /// Return `None` if the `at` position was wrong or `point`
    /// dimentions did not match the expected ones.
    fn to_univariate_at_point(
        &self,
        at: usize,
        point: &[F],
    ) -> Option<univariate::SparsePolynomial<F>>;

    /// Returns the number of variables in `self`
    fn num_vars(&self) -> usize;

    /// Returns a list of evaluations over the domain, which is the
    /// boolean hypercube.
    fn to_evaluations(&self) -> Vec<F>;
}

impl<F: Field> SumCheckPolynomial<F> for multivariate::SparsePolynomial<F, SparseTerm> {
    fn evaluate(&self, point: &[F]) -> Option<F> {
        Some(Polynomial::evaluate(self, &point.to_owned()))
    }

    fn to_univariate_at_point(
        &self,
        at: usize,
        point: &[F],
    ) -> Option<univariate::SparsePolynomial<F>> {
        let mut res = univariate::SparsePolynomial::zero();
        let mut point_temp = point.to_vec();
        point_temp[at] = F::one();

        for (coeff, term) in self.terms() {
            let eval = term.evaluate(&point_temp);
            let power = match term
                .vars()
                .iter()
                .zip(term.powers().iter())
                .find(|(&v, _)| v == at)
            {
                Some((_, p)) => *p,
                None => 0,
            };
            let new_coeff = *coeff * eval;
            res += &univariate::SparsePolynomial::from_coefficients_slice(&[(power, new_coeff)]);
        }
        Some(res)
    }

    fn num_vars(&self) -> usize {
        DenseMVPolynomial::num_vars(self)
    }

    fn to_evaluations(&self) -> Vec<F> {
        BooleanHypercube::new(DenseMVPolynomial::num_vars(self) as u32)
            .into_iter()
            .map(|point| Polynomial::evaluate(self, &point))
            .collect()
    }
}

/// The state of the Verifier.
pub struct Verifier<F: Field, P: SumCheckPolynomial<F>> {
    /// Number of variables in the original polynomial.
    n: usize,

    /// A $C_1$ value claimed by the Prover.
    c_1: F,

    /// Univariate polynomials $g_1,...,g_{j-1}$ received from the [`Prover`].
    g_part: Vec<univariate::SparsePolynomial<F>>,

    /// Previously picked random values $r_1,...,r_{j-1}$.
    r: Vec<F>,

    /// Original polynomial for oracle access
    g: P,
}

/// Values returned by Validator as a result of its run on every step.
pub enum VerifierRoundResult<F: Field> {
    /// On $j$-th round the verifier outputs a random $r_j$ value
    JthRound(F),

    /// On final round the verifier outputs `true` or `false` if it accepts
    /// or rejects the proof.
    FinalRound(bool),
}

impl<F: Field, P: SumCheckPolynomial<F>> Verifier<F, P> {
    /// Create the new state of the [`Verifier`].
    ///
    /// $n$ - degree of the polynomial
    /// $C_1$ - the value claimed to be true answer by the [`Prover`].
    /// $g$ - the polynimial itself for oracle access by the [`Verifier`].
    pub fn new(n: usize, c_1: F, g: P) -> Self {
        Self {
            n,
            c_1,
            g_part: Vec::with_capacity(n),
            r: Vec::with_capacity(n),
            g,
        }
    }

    /// Perform the $j$-th round of the [`Verifier]` side of the protocol.
    ///
    /// $g_j$ - a univariate polynomial sent in this round by the [`Prover`].
    pub fn round<R: Rng>(
        &mut self,
        g_j: univariate::SparsePolynomial<F>,
        rng: &mut R,
    ) -> Result<VerifierRoundResult<F>, Error> {
        let r_j = F::rand(rng);
        if self.r.is_empty() {
            // First Round
            let evaluation = g_j.evaluate(&F::zero()) + g_j.evaluate(&F::one());
            if self.c_1 != evaluation {
                Err(Error::ProverClaimMismatch(
                    format!("{}", self.c_1),
                    format!("{}", evaluation),
                ))
            } else {
                self.g_part.push(g_j);
                self.r.push(r_j);

                Ok(VerifierRoundResult::JthRound(r_j))
            }
        } else if self.r.len() == (self.n - 1) {
            // Last round
            self.r.push(r_j);
            Ok(VerifierRoundResult::FinalRound(
                g_j.evaluate(&r_j) == self.g.evaluate(&self.r).unwrap(),
            ))
        } else {
            // j-th round
            let g_jprev = self.g_part.last().unwrap();
            let r_jprev = self.r.last().unwrap();

            let prev_evaluation = g_jprev.evaluate(r_jprev);
            let evaluation = g_j.evaluate(&F::zero()) + g_j.evaluate(&F::one());
            if prev_evaluation != evaluation {
                return Err(Error::ProverClaimMismatch(
                    format!("{}", prev_evaluation),
                    format!("{}", evaluation),
                ));
            }

            self.g_part.push(g_j);
            self.r.push(r_j);

            Ok(VerifierRoundResult::JthRound(r_j))
        }
    }
}

/// Partially evaluate multivariate polynomial $g$ to be a univariate polynimial
/// of var $X_j$.
///
/// r: values $r_1 ... r_{j-1}$ that variables $X_1,...,X_{j-1}$ have been bound to.
///
/// $$
/// \sum_{(x_{j+1},...,x_\nu) \in \lbrace 0, 1 \rbrace ^ {\nu - 1}}
/// g(r_1,...,r_{j-1},X_j,x_{j+1},...,x_n)
/// $$
fn multivariate_to_univariate_with_fixed_vars<F: Field, P: SumCheckPolynomial<F>>(
    g: &P,
    r: &[F],
    j: usize,
) -> Option<univariate::SparsePolynomial<F>> {
    let mut res = univariate::SparsePolynomial::<F>::zero();

    // A Boolean hypercube over variables X_{j+1}...X_{n}.
    for x_point in BooleanHypercube::<F>::new((g.num_vars() - j - 1) as u32) {
        // [r_1,...,r_{j-1},1,X_{j+1},...,X_n]
        let mut point = r.to_vec();
        point.push(F::one());
        point.extend(x_point.into_iter());

        let r = g.to_univariate_at_point(j, &point)?;
        res += &r;
    }

    Some(res)
}

#[cfg(test)]
mod tests {
    use ark_ff::{
        fields::Fp64,
        fields::{MontBackend, MontConfig},
        Field, One, PrimeField, UniformRand,
    };
    use ark_poly::{
        multivariate::{self, SparseTerm, Term},
        DenseMVPolynomial, Polynomial,
    };
    use ark_std::{rand::Rng, test_rng};
    use pretty_assertions::assert_eq;

    use crate::{Prover, SumCheckPolynomial, Verifier, VerifierRoundResult};

    #[derive(MontConfig)]
    #[modulus = "5"]
    #[generator = "2"]
    struct FrConfig;

    type Fp5 = Fp64<MontBackend<FrConfig, 1>>;

    /// Generate random `l`-variate polynomial of maximum individual degree `d`
    fn rand_poly<R: Rng, F: Field>(
        l: usize,
        d: usize,
        rng: &mut R,
    ) -> multivariate::SparsePolynomial<F, SparseTerm> {
        let mut random_terms = Vec::new();
        let num_terms = rng.gen_range(1..1000);
        // For each term, randomly select up to `l` variables with degree
        // in [1,d] and random coefficient
        random_terms.push((F::rand(rng), SparseTerm::new(vec![])));
        for _ in 1..num_terms {
            let term = (0..l)
                .map(|i| {
                    if rng.gen_bool(0.5) {
                        Some((i, rng.gen_range(1..(d + 1))))
                    } else {
                        None
                    }
                })
                .filter(|t| t.is_some())
                .map(|t| t.unwrap())
                .collect();
            let coeff = F::rand(rng);
            random_terms.push((coeff, SparseTerm::new(term)));
        }
        multivariate::SparsePolynomial::from_coefficients_slice(l, &random_terms)
    }

    #[test]
    fn basic_test() {
        // 2 * x_1 * x_2 + 3 * x_1^2 * x_2^2
        let poly = multivariate::SparsePolynomial::from_coefficients_slice(
            2,
            &[
                (
                    Fp5::from_bigint(2u32.into()).unwrap(),
                    multivariate::SparseTerm::new(vec![(0, 1), (1, 1)]),
                ),
                (
                    Fp5::from_bigint(3u32.into()).unwrap(),
                    multivariate::SparseTerm::new(vec![(0, 2), (1, 2)]),
                ),
            ],
        );

        let res = poly.to_univariate_at_point(0, &[2u32.into(), 1u32.into()]);

        println!("res {:?}", res);
    }

    #[test]
    fn randomized_test() {
        let rng = &mut test_rng();

        for var_count in 1..10 {
            let degree = 10;
            let poly: multivariate::SparsePolynomial<Fp5, SparseTerm> =
                rand_poly(var_count, degree, rng);

            let mut point = Vec::with_capacity(var_count);

            for _ in 0..var_count {
                point.push(Fp5::rand(rng));
            }

            let normal_evaluation = Polynomial::evaluate(&poly, &point);

            for fixed_var_idx in 0..var_count {
                let reduced_uni_poly = poly.to_univariate_at_point(fixed_var_idx, &point).unwrap();

                let univariate_evaluation = reduced_uni_poly.evaluate(&point[fixed_var_idx]);

                assert_eq!(
                    normal_evaluation, univariate_evaluation,
                    "{:?} \n\n {:?}",
                    poly, reduced_uni_poly
                );
            }
        }
    }

    #[test]
    fn protocol_test() {
        for n in 2..10 {
            let rng = &mut test_rng();

            let g = rand_poly(n, 3, rng);

            let mut prover = Prover::new(g.clone());
            let c_1 = prover.c_1();
            let mut r_j = Fp5::one();
            let mut verifier = Verifier::new(n, c_1, g);

            for j in 0..n {
                let g_j = prover.round(r_j, j).unwrap();
                let verifier_res = verifier.round(g_j, rng).unwrap();
                match verifier_res {
                    VerifierRoundResult::JthRound(r) => {
                        r_j = r;
                    }
                    VerifierRoundResult::FinalRound(res) => {
                        assert!(res);
                        break;
                    }
                }
            }
        }
    }
}
