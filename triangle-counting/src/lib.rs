use ark_ff::{FftField, Field};
use ark_poly::{
    univariate::{DensePolynomial, SparsePolynomial},
    DenseMultilinearExtension, EvaluationDomain, Evaluations, GeneralEvaluationDomain,
    MultilinearExtension,
};
use sum_check_protocol::SumCheckPolynomial;

#[derive(Clone)]
pub struct G<F: Field> {
    f_a: DenseMultilinearExtension<F>,
}

impl<F: Field> G<F> {
    pub fn new_adj_matrix<M>(num_vars: usize, matrix: M) -> Self
    where
        M: IntoIterator<Item = bool>,
    {
        let g = DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            matrix
                .into_iter()
                .map(|b| if b { F::one() } else { F::zero() })
                .collect(),
        );

        Self { f_a: g }
    }
}

impl<F: FftField> G<F> {
    fn g_to_univariate_at(&self, at: usize, point: &[F]) -> DensePolynomial<F> {
        let mut fixed_1 = self.f_a.fix_variables(&point[..at]);

        if at != self.f_a.num_vars() - 1 {
            fixed_1.relabel_inplace(0, fixed_1.num_vars() - 1, 1);
            fixed_1 = fixed_1.fix_variables(&[point[point.len() - 1]]);
            let fixed_2 = fixed_1.fix_variables(&point[at + 1..point.len() - 1]);
            fixed_1 = fixed_2;
        }

        let domain = GeneralEvaluationDomain::new(3).unwrap();

        let evaluations = domain
            .elements()
            .map(|e| fixed_1.evaluate(&[e]).unwrap())
            .collect();

        let evaluations = Evaluations::from_vec_and_domain(evaluations, domain);

        evaluations.interpolate()
    }
}

impl<F: FftField> SumCheckPolynomial<F> for G<F> {
    /// Evaluate over a point $(X, Y, Z)$.
    fn evaluate(&self, point: &[F]) -> Option<F> {
        assert!(point.len() == (self.f_a.num_vars() / 2) * 3);

        let mut x_z = point[..self.f_a.num_vars() / 2].to_owned();
        x_z.extend_from_slice(&point[self.f_a.num_vars()..]);
        // X, Y
        Some(
            self.f_a.evaluate(&point[..self.f_a.num_vars()])? *
            // Y, Z
            self.f_a.evaluate(&point[self.f_a.num_vars() / 2 .. ])?
            * self.f_a.evaluate(&x_z)?,
        )
    }

    fn to_univariate_at_point(&self, at: usize, point: &[F]) -> Option<SparsePolynomial<F>> {
        let x_y = &point[..self.f_a.num_vars()];
        let y_z = &point[self.f_a.num_vars() / 2..];
        let mut x_z = point[..self.f_a.num_vars() / 2].to_owned();
        x_z.extend_from_slice(&point[self.f_a.num_vars()..]);

        match at / (self.f_a.num_vars() / 2) {
            0 => {
                let f_y_z = self.f_a.evaluate(y_z).unwrap();

                let a = self.g_to_univariate_at(at, x_y);
                let b = self.g_to_univariate_at(at, &x_z);
                Some((&(&a * &b) * f_y_z).into())
            }
            1 => {
                let f_x_z = self.f_a.evaluate(&x_z).unwrap();

                let a = self.g_to_univariate_at(at, x_y);
                let b = self.g_to_univariate_at(at - self.f_a.num_vars() / 2, y_z);
                Some((&(&a * &b) * f_x_z).into())
            }
            2 => {
                let f_x_y = self.f_a.evaluate(x_y).unwrap();

                let a = self.g_to_univariate_at(at - self.f_a.num_vars() / 2, &x_z);
                let b = self.g_to_univariate_at(at - self.f_a.num_vars() / 2, y_z);

                Some((&(&a * &b) * f_x_y).into())
            }
            _ => None,
        }
    }

    fn num_vars(&self) -> usize {
        (self.f_a.num_vars() / 2) * 3
    }

    fn to_evaluations(&self) -> Vec<F> {
        let x_size = self.f_a.num_vars() / 2;
        let bit_mask_least_significant = (usize::MAX << x_size) ^ usize::MAX;
        let bit_mask_most_significant = bit_mask_least_significant << x_size;

        let mut res = vec![F::zero(); 2usize.pow(self.num_vars() as u32)];
        let evaluations = self.f_a.to_evaluations();

        for (x_y, evaluation) in evaluations.iter().enumerate() {
            for z in 0..2usize.pow(x_size as u32) {
                let f_x_y = evaluation;
                let f_y_z_idx = ((x_y << x_size) & bit_mask_most_significant) | z;
                let f_y_z = evaluations[f_y_z_idx];

                let f_x_z_idx = (x_y & bit_mask_most_significant) | z;
                let f_x_z = evaluations[f_x_z_idx];

                let f_x_y_z = (*f_x_y * f_y_z) * f_x_z;

                res[(x_y << x_size) | z] = f_x_y_z;
            }
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{Fp64, MontBackend, MontConfig, One, PrimeField};
    use ark_std::{rand::Rng, test_rng};
    use pretty_assertions::assert_eq;
    use sum_check_protocol::{Prover, Verifier, VerifierRoundResult};

    use super::*;

    struct AdjMatrix(Vec<Vec<bool>>);

    impl AdjMatrix {
        fn new<R: Rng>(n: usize, rng: &mut R) -> Self {
            let mut res = Vec::with_capacity(n);

            for _ in 0..n {
                res.push(vec![false; n]);
            }

            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let are_connected = rng.gen();

                        res[i][j] = are_connected;
                        res[j][i] = are_connected;
                    }
                }
            }

            Self(res)
        }

        fn triangle_count(&self) -> usize {
            let mut count = 0;
            let n = self.0.len();

            for x in 0..n {
                for y in 0..n {
                    for z in 0..n {
                        if self.0[x][y] & self.0[y][z] & self.0[x][z] {
                            count += 1;
                        }
                    }
                }
            }

            count / 6
        }
    }

    #[test]
    fn test_simple_matrix() {
        #[derive(MontConfig)]
        #[modulus = "389"]
        #[generator = "2"]
        struct FrConfig;

        type Fp389 = Fp64<MontBackend<FrConfig, 1>>;

        let rng = &mut test_rng();

        let adj_matrix = vec![
            vec![false, true, true, false],
            vec![true, false, true, false],
            vec![true, true, false, false],
            vec![false, false, false, false],
        ];

        let g: G<Fp389> =
            G::new_adj_matrix(adj_matrix.len(), adj_matrix.iter().flatten().map(|b| *b));

        let num_vars = g.num_vars();
        let mut prover = Prover::new(g.clone());
        let c_1 = prover.c_1();
        let mut r_j = Fp389::one();
        let mut verifier = Verifier::new(num_vars, c_1, g);

        for j in 0..num_vars {
            let g_j = prover.round(r_j, j).unwrap();
            let verifier_res = verifier.round(g_j, rng).unwrap();
            match verifier_res {
                VerifierRoundResult::JthRound(r) => {
                    r_j = r;
                }
                VerifierRoundResult::FinalRound(res) => {
                    assert!(res);
                    return;
                }
            }
        }

        panic!("should have returned on FinalRound from verifier");
    }

    #[test]
    fn randomized_test() {
        let rng = &mut test_rng();

        // TODO: This field is of the wrong size, p > 6 * n^3 needed
        #[derive(MontConfig)]
        #[modulus = "389"]
        #[generator = "2"]
        struct FrConfig;

        type Fp389 = Fp64<MontBackend<FrConfig, 1>>;

        for i in 2..8 {
            let n = 2usize.pow(i);

            let test_matrix = AdjMatrix::new(n, rng);

            let triangle_count = test_matrix.triangle_count();

            let g: G<Fp389> = G::new_adj_matrix(
                (f64::from(n as u32).log2() as usize) * 2,
                test_matrix.0.iter().flatten().map(|b| *b),
            );

            let mut prover = Prover::new(g.clone());
            let c_1 = prover.c_1();

            let count = c_1.into_bigint().as_ref()[0];

            assert_eq!(
                triangle_count * 6,
                count as usize,
                "mismatch for size {n}: {triangle_count} {count}"
            );

            let mut r_j = Fp389::one();
            let num_vars = g.num_vars();
            let mut verifier = Verifier::new(num_vars, c_1, g);

            for j in 0..num_vars {
                let g_j = prover.round(r_j, j).unwrap();
                let verifier_res = verifier.round(g_j, rng).unwrap();
                match verifier_res {
                    VerifierRoundResult::JthRound(r) => r_j = r,
                    VerifierRoundResult::FinalRound(res) => {
                        assert!(res);
                        return;
                    }
                }
            }

            panic!("Should have returned on the last round of Verifier");
        }
    }
}
