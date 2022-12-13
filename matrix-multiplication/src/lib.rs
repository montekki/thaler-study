use ark_ff::{FftField, Field};
use ark_poly::{univariate::SparsePolynomial, DenseMultilinearExtension, MultilinearExtension};
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

fn interpolate_quadratic_poly<F: Field>(points: &[(F, F)]) -> SparsePolynomial<F> {
    let denominator_1 = (points[0].0 - points[1].0) * (points[0].0 - points[2].0);
    let denominator_2 = (points[1].0 - points[0].0) * (points[1].0 - points[2].0);
    let denominator_3 = (points[2].0 - points[0].0) * (points[2].0 - points[1].0);

    let mut coeffs_1 = vec![
        (0, points[1].0 * points[2].0),
        (1, -points[1].0 - points[2].0),
        (2, F::one()),
    ];

    for item in &mut coeffs_1 {
        item.1 *= points[0].1;
        item.1 /= denominator_1;
    }

    let mut coeffs_2 = vec![
        (0, points[0].0 * points[2].0),
        (1, -points[0].0 - points[2].0),
        (2, F::one()),
    ];

    for item in &mut coeffs_2 {
        item.1 *= points[1].1;
        item.1 /= denominator_2;
    }

    let mut coeffs_3 = vec![
        (0, points[0].0 * points[1].0),
        (1, -points[0].0 - points[1].0),
        (2, F::one()),
    ];

    for item in &mut coeffs_3 {
        item.1 *= points[2].1;
        item.1 /= denominator_3;
    }

    let poly_1 = SparsePolynomial::from_coefficients_vec(coeffs_1);
    let poly_2 = SparsePolynomial::from_coefficients_vec(coeffs_2);
    let poly_3 = SparsePolynomial::from_coefficients_vec(coeffs_3);

    poly_1 + poly_2 + poly_3
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
        let two = F::one() + F::one();
        let mut evals = [F::zero(); 3];

        for i in 0..2usize.pow(self.num_vars() as u32) {
            if i & 1 == 1 {
                evals[1] += self.f_a[i] * self.f_b[i];
                evals[2] +=
                    (two * self.f_a[i] - self.f_a[i - 1]) * (two * self.f_b[i] - self.f_b[i - 1]);
            } else {
                evals[0] += self.f_a[i] * self.f_b[i];
            }
        }

        let points = vec![
            (F::zero(), evals[0]),
            (F::one(), evals[1]),
            (F::one() + F::one(), evals[2]),
        ];

        interpolate_quadratic_poly(&points)
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

                let mut verifier = Verifier::new(num_vars, Some(g));
                verifier.set_c_1(c_1);
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

                    let mut prover = Prover::new(g.clone());

                    let c_1 = prover.c_1();
                    assert_eq!(c_1, c.0[i as usize][j as usize]);

                    let mut resu = Fp5::zero();

                    // Check that `evaluate` works correctly and returns
                    // expected results.
                    for a in 0..n as u32 {
                        let point: Vec<_> = u32_to_boolean_vec(a, p as usize);

                        resu += g.evaluate(&point).unwrap();
                    }

                    assert_eq!(c_1, resu);

                    let num_vars = g.num_vars();

                    let mut r_j = Fp5::one();
                    let mut verifier = Verifier::new(num_vars, Some(g));
                    verifier.set_c_1(c_1);

                    for j in 0..num_vars {
                        let g_j = prover.round(r_j, j).unwrap();
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
    }
}
