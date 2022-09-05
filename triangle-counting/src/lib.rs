use std::cmp;

use ark_ff::{FftField, Field};
use ark_poly::{
    univariate::SparsePolynomial, DenseMultilinearExtension, EvaluationDomain, Evaluations,
    GeneralEvaluationDomain, MultilinearExtension,
};
use sum_check_protocol::SumCheckPolynomial;

/// A polynomial
///
/// $$
/// g(X,Y,Z) = \tilde{f}_A(X,Y) \cdot \tilde{f}_A(Y,Z) \cdot \tilde{f}_A(X,Z)
/// $$
///
/// that when used with Sum Check protocol yields an IP for computing a $6\Delta$
/// number of triangles in a graph.
///
/// Holds three copies of a multilinear extension $\tilde{f}_A$ since fixing
/// a variable in $g$ will lead to different multilinear equations.
#[derive(Clone)]
pub struct G<F: Field> {
    f_a_1: DenseMultilinearExtension<F>,
    f_a_2: DenseMultilinearExtension<F>,
    f_a_3: DenseMultilinearExtension<F>,
    var_len: usize,
}

impl<F: Field> G<F> {
    /// Creates a new $3 \log n$-variate polynomial $g(X,Y,Z)$ from
    /// a graph adjacency matrix.
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

        let var_len = g.num_vars() / 2;
        Self {
            f_a_1: g.clone(),
            f_a_2: g.clone(),
            f_a_3: g,
            var_len,
        }
    }

    fn x_vars_num(&self) -> usize {
        self.f_a_1.num_vars().saturating_sub(self.var_len)
    }

    fn y_vars_num(&self) -> usize {
        self.f_a_2.num_vars().saturating_sub(self.var_len)
    }

    fn z_vars_num(&self) -> usize {
        if self.f_a_3.num_vars() < self.var_len {
            self.f_a_3.num_vars()
        } else {
            self.var_len
        }
    }
}

impl<F: FftField> SumCheckPolynomial<F> for G<F> {
    fn evaluate(&self, point: &[F]) -> Option<F> {
        let x_y_point = point
            .get(..self.x_vars_num() + self.y_vars_num())
            .unwrap_or(&[]);

        let y_z_point = point.get(self.x_vars_num()..).unwrap_or(&[]);
        let f_x_y_evaluation = self.f_a_1.evaluate(x_y_point)?;
        let f_y_z_evaluation = self.f_a_2.evaluate(y_z_point)?;

        let mut x_z = point.get(..self.x_vars_num()).unwrap_or(&[]).to_owned();

        x_z.extend_from_slice(&point[self.x_vars_num() + self.y_vars_num()..]);

        let f_x_z_evaluation = self.f_a_3.evaluate(&x_z)?;

        Some(f_x_y_evaluation * f_x_z_evaluation * f_y_z_evaluation)
    }

    fn fix_variables(&self, partial_point: &[F]) -> Self {
        let x_y_point = partial_point
            .get(..cmp::min(self.x_vars_num() + self.y_vars_num(), partial_point.len()))
            .unwrap_or(&[]);

        let y_z_point = &partial_point.get(self.x_vars_num()..).unwrap_or(&[]);

        let mut x_z_point = partial_point
            .get(..cmp::min(self.x_vars_num(), partial_point.len()))
            .unwrap_or(&[])
            .to_owned();

        x_z_point.extend_from_slice(
            partial_point
                .get(self.x_vars_num() + self.y_vars_num()..)
                .unwrap_or(&[]),
        );

        let f_a_1 = self.f_a_1.fix_variables(x_y_point);
        let f_a_2 = self.f_a_2.fix_variables(y_z_point);
        let f_a_3 = self.f_a_3.fix_variables(&x_z_point);
        let var_len = self.var_len;

        Self {
            f_a_1,
            f_a_2,
            f_a_3,
            var_len,
        }
    }

    fn to_univariate(&self) -> SparsePolynomial<F> {
        let domain = GeneralEvaluationDomain::new(3).unwrap();

        let evals = domain
            .elements()
            .map(|e| self.fix_variables(&[e]).to_evaluations().into_iter().sum())
            .collect();

        let evaluations = Evaluations::from_vec_and_domain(evals, domain);
        let p = evaluations.interpolate();

        p.into()
    }

    fn num_vars(&self) -> usize {
        self.x_vars_num() + self.y_vars_num() + self.z_vars_num()
    }

    fn to_evaluations(&self) -> Vec<F> {
        // combine the evaluations of separate multilinear
        // extensions into a vector of evaluations of the
        // whole polynomial
        let f_a_1_evals = self.f_a_1.to_evaluations();
        let f_a_2_evals = self.f_a_2.to_evaluations();
        let f_a_3_evals = self.f_a_3.to_evaluations();

        let mut res = vec![];
        for x_idx in 0..2usize.pow(self.x_vars_num() as u32) {
            for y_idx in 0..2usize.pow(self.y_vars_num() as u32) {
                for z_idx in 0..2usize.pow(self.z_vars_num() as u32) {
                    // indexing into evaluations in this order: x, y, z
                    // and so when combining for example x and y
                    // x always occupies the lower bits and y occupies the
                    // higher bits in the index.

                    let idx_1 = idx(y_idx, x_idx, self.x_vars_num());
                    let idx_2 = idx(z_idx, y_idx, self.y_vars_num());
                    let idx_3 = idx(z_idx, x_idx, self.x_vars_num());

                    res.push(f_a_1_evals[idx_1] * f_a_2_evals[idx_2] * f_a_3_evals[idx_3])
                }
            }
        }

        res
    }
}

/// Combine indices of two variables into one to be able
/// to index into evaluations of polynomial.
fn idx(i: usize, j: usize, num_vars: usize) -> usize {
    (i << num_vars) | j
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

        #[derive(MontConfig)]
        #[modulus = "1572869"]
        #[generator = "2"]
        struct FrConfig;

        type Fp389 = Fp64<MontBackend<FrConfig, 1>>;

        for i in 1..8 {
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
                    }
                }
            }
        }
    }
}
