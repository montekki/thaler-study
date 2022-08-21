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
        assert!(point.len() * 2 == self.f_a.num_vars() * 3);

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
        self.f_a.to_evaluations()
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{Fp64, MontBackend, MontConfig, One};
    use ark_std::test_rng;
    use sum_check_protocol::{Prover, Verifier, VerifierRoundResult};

    use super::*;

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
}
