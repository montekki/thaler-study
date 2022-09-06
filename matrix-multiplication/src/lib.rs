use ark_ff::{FftField, Field};
use ark_poly::{
    univariate::SparsePolynomial, DenseMultilinearExtension, EvaluationDomain, Evaluations,
    GeneralEvaluationDomain, MultilinearExtension,
};
use bitvec::{slice::BitSlice, vec::BitVec};
use multilinear_extensions::lagrange_basis_poly_at;
use sum_check_protocol::SumCheckPolynomial;

/// A polynomial of form
/// $g(z) = \tilde{f}_A(r_1,z) \cdot \tilde{f}_B(z, r_2)$.
/// used to apply Sum Check protocol
/// to matrix multiplication
#[derive(Clone)]
pub struct G<F: Field> {
    f_a: DenseMultilinearExtension<F>,
    f_b: DenseMultilinearExtension<F>,
}

impl<F: Field> G<F> {
    /// Create $g$ for evaluating $f_A \cdot f_B$ at any given
    /// point $(r_1, r_2) \in \mathbb{F}^{\log n \times \log n}$.
    /// The polynomial is of the form
    /// $g(z) = \tilde{f}_A(r_1,z) \cdot \tilde{f}_B(z, r_2)$.
    ///
    /// # Arguments
    ///
    /// `n` - Number of variables in the polynomial.
    ///
    /// `a` - Iterator over the elements of square matrix $A$
    ///
    /// `b` - Iterator over the elements of Square matrix $B$
    ///
    /// `point` - Point $(r_1, r_2)$.
    pub fn new<M>(n: usize, a: M, b: M, point: &[F]) -> Self
    where
        M: IntoIterator<Item = F>,
    {
        let f_a = DenseMultilinearExtension::from_evaluations_vec(n * 2, a.into_iter().collect());
        let f_a = f_a.relabel(0, n, n);
        let f_a = f_a.fix_variables(&point[..n]);

        let f_b = DenseMultilinearExtension::from_evaluations_vec(n * 2, b.into_iter().collect());
        let f_b = f_b.fix_variables(&point[n..]);

        assert_eq!(f_a.num_vars(), n);
        assert_eq!(f_b.num_vars(), n);

        Self { f_a, f_b }
    }
}

impl<F: FftField> SumCheckPolynomial<F> for G<F> {
    fn evaluate(&self, point: &[F]) -> Option<F> {
        let f_a = self.f_a.evaluate(point)?;
        let f_b = self.f_b.evaluate(point)?;

        Some(f_a * f_b)
    }

    fn fix_variables(&self, partial_point: &[F]) -> Self {
        let f_a = self.f_a.fix_variables(partial_point);
        let f_b = self.f_b.fix_variables(partial_point);

        Self { f_a, f_b }
    }

    fn to_univariate(&self) -> SparsePolynomial<F> {
        let domain: GeneralEvaluationDomain<F> = GeneralEvaluationDomain::new(3).unwrap();

        let evals = domain
            .elements()
            .map(|e| {
                let f_a_evals = self.f_a.fix_variables(&[e]).to_evaluations();
                let f_b_evals = self.f_b.fix_variables(&[e]).to_evaluations();
                f_a_evals
                    .into_iter()
                    .zip(f_b_evals.into_iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();

        let evaluations = Evaluations::from_vec_and_domain(evals, domain);
        let p = evaluations.interpolate();

        p.into()
    }

    fn num_vars(&self) -> usize {
        self.f_a.num_vars()
    }

    fn to_evaluations(&self) -> Vec<F> {
        let mut f_a_evals = self.f_a.to_evaluations();
        let f_b_evals = self.f_b.to_evaluations();

        for (i, eval) in f_b_evals.into_iter().enumerate() {
            f_a_evals[i] *= eval
        }

        f_a_evals
    }
}

#[derive(Clone)]
pub struct G2<F: Field> {
    r_1: usize,
    r_2: usize,
    a: Vec<Vec<F>>,
    b: Vec<Vec<F>>,
}

impl<F: Field> G2<F> {
    pub fn new(r_1: usize, r_2: usize, a: Vec<Vec<F>>, b: Vec<Vec<F>>) -> Self {
        Self { r_1, r_2, a, b }
    }
}

fn u32_to_boolean_vec<F: Field>(v: u32, bits: usize) -> Vec<F> {
    let slice = v.to_le_bytes();
    let bitslice: &BitSlice<u8> = BitSlice::try_from_slice(&slice).unwrap();
    bitslice
        .iter()
        .take(bits)
        .map(|bit| match *bit {
            true => F::one(),
            false => F::zero(),
        })
        .collect()
}

fn point_to_usize<F: Field>(point: &[F]) -> Option<usize> {
    let bitvec: BitVec = point
        .iter()
        .map(|&p| {
            if p == F::one() {
                true
            } else if p == F::zero() {
                false
            } else {
                panic!("expected 0 or 1 got {:?}", p)
            }
        })
        .collect();

    bitvec.into_vec().get(0).map(|f| *f)
}

impl<F: FftField> SumCheckPolynomial<F> for G2<F> {
    fn evaluate(&self, point: &[F]) -> Option<F> {
        let mut a_eval = F::zero();
        let mut b_eval = F::zero();
        let n = self.num_vars();

        let boolean_suffix_len = point
            .iter()
            .rev()
            .take_while(|&&c| c == F::zero() || c == F::one())
            .count();

        let mask = point_to_usize(&point[point.len() - boolean_suffix_len..]).unwrap_or_else(|| 0);

        for i in 0..self.a.len() {
            let i_shifted = i >> (point.len() - boolean_suffix_len);
            if i_shifted == mask {
                let w_1 = u32_to_boolean_vec(i as u32, n);

                let w_2: Vec<F> = u32_to_boolean_vec(i as u32, n);
                let lagr_basis_a = lagrange_basis_poly_at(point, &w_1).unwrap();
                let lagr_basis_b = lagrange_basis_poly_at(point, &w_2).unwrap();

                a_eval += self.a[self.r_1][i] * lagr_basis_a;
                b_eval += self.b[i][self.r_2] * lagr_basis_b;
            }
        }

        Some(a_eval * b_eval)
    }

    fn to_univariate_at_point(
        &self,
        at: usize,
        point: &[F],
    ) -> Option<ark_poly::univariate::SparsePolynomial<F>> {
        let mut point = point.to_vec();
        let domain = GeneralEvaluationDomain::new(3).unwrap();

        let evaluations: Vec<F> = domain
            .elements()
            .map(|e| {
                point[at] = e;
                self.evaluate(&point).unwrap()
            })
            .collect();

        let evaluations = Evaluations::from_vec_and_domain(evaluations, domain);

        Some(evaluations.interpolate().into())
    }

    fn num_vars(&self) -> usize {
        f64::from(self.a.len() as u32).log2() as usize
    }

    fn to_evaluations(&self) -> Vec<F> {
        let mut point = u32_to_boolean_vec(self.r_1 as u32, self.num_vars());
        point.append(&mut u32_to_boolean_vec(self.r_2 as u32, self.num_vars()));

        let g = G::new(
            self.num_vars() * 2,
            self.a.iter().flatten().map(Clone::clone),
            self.b.iter().flatten().map(Clone::clone),
            &point,
        );

        g.to_evaluations()
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Mul;

    use ark_ff::{Fp64, MontBackend, MontConfig, One, PrimeField, Zero};
    use ark_std::{rand::Rng, test_rng};
    use bitvec::slice::BitSlice;
    use pretty_assertions::assert_eq;
    use sum_check_protocol::{Prover, Verifier, VerifierRoundResult};

    use super::*;
    #[derive(MontConfig)]
    #[modulus = "5"]
    #[generator = "2"]
    struct FrConfig;

    type Fp5 = Fp64<MontBackend<FrConfig, 1>>;

    #[derive(Debug, Clone, PartialEq)]
    struct Matrix<F: std::fmt::Debug + Field>(Vec<Vec<F>>);

    impl<F: std::fmt::Debug + Field> Matrix<F> {
        /// Generate a random $n times n$ matrix of `F`.
        fn new<R: Rng>(n: usize, rng: &mut R) -> Self {
            Self(
                (0..n)
                    .into_iter()
                    .map(|_| (0..n).into_iter().map(|_| F::rand(rng)).collect())
                    .collect(),
            )
        }
    }

    impl<F: Field> Mul for Matrix<F> {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            let mut res = self.clone();
            let n = res.0.len();

            for i in 0..n {
                for j in 0..n {
                    res.0[i][j] = F::zero();

                    for k in 0..n {
                        res.0[i][j] += self.0[i][k] * rhs.0[k][j];
                    }
                }
            }

            res
        }
    }

    #[test]
    fn matrix_test_from_book() {
        let a = vec![
            vec![
                Fp5::from_bigint(0u32.into()).unwrap(),
                Fp5::from_bigint(1u32.into()).unwrap(),
            ],
            vec![
                Fp5::from_bigint(2u32.into()).unwrap(),
                Fp5::from_bigint(0u32.into()).unwrap(),
            ],
        ];

        let b = vec![
            vec![
                Fp5::from_bigint(1u32.into()).unwrap(),
                Fp5::from_bigint(0u32.into()).unwrap(),
            ],
            vec![
                Fp5::from_bigint(0u32.into()).unwrap(),
                Fp5::from_bigint(4u32.into()).unwrap(),
            ],
        ];

        let c = vec![
            vec![
                Fp5::from_bigint(0u32.into()).unwrap(),
                Fp5::from_bigint(4u32.into()).unwrap(),
            ],
            vec![
                Fp5::from_bigint(2u32.into()).unwrap(),
                Fp5::from_bigint(0u32.into()).unwrap(),
            ],
        ];

        let a = Matrix(a);
        let b = Matrix(b);

        let c = Matrix(c);

        assert_eq!(c, a * b);
    }

    #[test]
    fn example_from_book() {
        let rng = &mut test_rng();
        let a = vec![
            vec![
                Fp5::from_bigint(0u32.into()).unwrap(),
                Fp5::from_bigint(1u32.into()).unwrap(),
            ],
            vec![
                Fp5::from_bigint(2u32.into()).unwrap(),
                Fp5::from_bigint(0u32.into()).unwrap(),
            ],
        ];

        let b = vec![
            vec![
                Fp5::from_bigint(1u32.into()).unwrap(),
                Fp5::from_bigint(0u32.into()).unwrap(),
            ],
            vec![
                Fp5::from_bigint(0u32.into()).unwrap(),
                Fp5::from_bigint(4u32.into()).unwrap(),
            ],
        ];

        for i in 0..2u32 {
            for j in 0..2u32 {
                let mut point: Vec<_> = u32_to_boolean_vec(i, 1usize);
                point.append(&mut u32_to_boolean_vec(j, 1usize));

                let g = G::new(
                    1,
                    a.iter().flatten().cloned(),
                    b.iter().flatten().cloned(),
                    &point,
                );

                let mut prover = Prover::new(g.clone());
                let c_1 = prover.c_1();

                let num_vars = g.num_vars();
                let mut r_j = Fp5::one();

                let mut verifier = Verifier::new(num_vars, c_1, g);
                for k in 0..num_vars {
                    let g_j = prover.round(r_j, k).unwrap();
                    let verifier_res = verifier.round(g_j, rng).unwrap();

                    match verifier_res {
                        VerifierRoundResult::JthRound(r) => r_j = r,
                        VerifierRoundResult::FinalRound(res) => {
                            assert!(res);
                        }
                    }
                }
            }
        }
    }

    fn u32_to_boolean_vec<F: Field>(v: u32, bits: usize) -> Vec<F> {
        let slice = v.to_le_bytes();
        let bitslice: &BitSlice<u8> = BitSlice::try_from_slice(&slice).unwrap();
        bitslice
            .iter()
            .take(bits)
            .map(|bit| match *bit {
                true => F::one(),
                false => F::zero(),
            })
            .collect()
    }

    #[test]
    fn randomized_test() {
        let rng = &mut test_rng();

        for p in 2..6 {
            let n = 2usize.pow(p);

            let a: Matrix<Fp5> = Matrix::new(n, rng);
            let b: Matrix<Fp5> = Matrix::new(n, rng);
            let c = a.clone() * b.clone();

            for i in 0..n as u32 {
                for j in 0..n as u32 {
                    let mut point: Vec<_> = u32_to_boolean_vec(i, p as usize);
                    point.append(&mut u32_to_boolean_vec(j, p as usize));
                    let g = G::new(
                        p as usize,
                        a.0.iter().flatten().cloned(),
                        b.0.iter().flatten().cloned(),
                        &point,
                    );

                    let g_2 = G2::new(i as usize, j as usize, a.0.clone(), b.0.clone());

                    let mut prover = Prover::new(g.clone());
                    let mut prover_2 = Prover::new(g_2.clone());

                    let c_1 = prover.c_1();
                    let c_1_2 = prover_2.c_1();

                    assert_eq!(c_1, c.0[i as usize][j as usize]);
                    assert_eq!(c_1, c_1_2);

                    let mut resu = Fp5::zero();
                    let mut resu_2 = Fp5::zero();

                    // Check that `evaluate` works correctly and returns
                    // expected results.
                    for a in 0..n as u32 {
                        let point: Vec<_> = u32_to_boolean_vec(a, p as usize);

                        resu += g.evaluate(&point).unwrap();
                        resu_2 += g_2.evaluate(&point).unwrap();
                    }

                    assert_eq!(c_1, resu);
                    assert_eq!(c_1, resu_2);

                    let num_vars = g.num_vars();
                    assert_eq!(num_vars, g_2.num_vars());

                    let mut r_j = Fp5::one();
                    let mut r_j_2 = Fp5::one();
                    let mut verifier = Verifier::new(num_vars, c_1, g);
                    let mut verifier_2 = Verifier::new(num_vars, c_1_2, g_2);

                    for j in 0..num_vars {
                        let g_j = prover.round(r_j, j).unwrap();
                        let g_j_2 = prover_2.round(r_j_2, j).unwrap();
                        let verifier_res = verifier.round(g_j, rng).unwrap();
                        let verifier_2_res = verifier_2.round(g_j_2, rng).unwrap();
                        match verifier_res {
                            VerifierRoundResult::JthRound(r) => r_j = r,
                            VerifierRoundResult::FinalRound(res) => {
                                assert!(res);
                            }
                        }
                        match verifier_2_res {
                            VerifierRoundResult::JthRound(r) => r_j_2 = r,
                            VerifierRoundResult::FinalRound(res) => {
                                assert!(res);
                            }
                        }
                    }
                }
            }
        }
    }
}
