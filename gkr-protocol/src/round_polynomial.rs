//! The polynomial $f^{(i)}_{r_i}$ used to run Sum-Check
//! at the $i$-th step of the GKR protocol.

use std::cmp;

use ark_ff::{FftField, Field};
use ark_poly::{
    univariate, DenseMultilinearExtension, EvaluationDomain, Evaluations, GeneralEvaluationDomain,
    MultilinearExtension,
};
use sum_check_protocol::SumCheckPolynomial;

/// A $2k_{i+1}$ variate polynomial used for each step of GKR protocol.
///
/// $$
/// f^{i}_{r_i}(b, c) \coloneqq
/// \widetilde{add}_i(r_i, b, c)(\tilde{W}\_{i+1}(b) +
/// \tilde{W}\_{i+1}(c)) +
/// \widetilde{mul}_i(r_i, b, c)(\tilde{W}\_{i+1}(b) \cdot
/// \tilde{W}\_{i+1}(c))
/// $$
#[derive(Clone)]
pub struct W<F: Field> {
    add_i: DenseMultilinearExtension<F>,
    mul_i: DenseMultilinearExtension<F>,
    w_b: DenseMultilinearExtension<F>,
    w_c: DenseMultilinearExtension<F>,
}

impl<F: Field> W<F> {
    /// Create a new `W` polynomial.
    pub fn new(
        add_i: DenseMultilinearExtension<F>,
        mul_i: DenseMultilinearExtension<F>,
        w_b: DenseMultilinearExtension<F>,
        w_c: DenseMultilinearExtension<F>,
    ) -> Self {
        Self {
            add_i,
            mul_i,
            w_b,
            w_c,
        }
    }
}

impl<F: FftField> SumCheckPolynomial<F> for W<F> {
    fn evaluate(&self, point: &[F]) -> Option<F> {
        let (b, c) = point.split_at({
            let ref this = self.w_b;
            this.num_vars
        });
        let add_e = self.add_i.evaluate(point)?;
        let mul_e = self.mul_i.evaluate(point)?;

        let w_b = self.w_b.evaluate(b)?;
        let w_c = self.w_c.evaluate(c)?;

        Some(add_e * (w_b + w_c) + mul_e * (w_b * w_c))
    }

    fn fix_variables(&self, partial_point: &[F]) -> Self {
        let b_partial = partial_point
            .get(..cmp::min(self.w_b.num_vars(), partial_point.len()))
            .unwrap_or(&[]);
        let c_partial = partial_point.get(self.w_b.num_vars()..).unwrap_or(&[]);

        let add_i = self.add_i.fix_variables(partial_point);
        let mul_i = self.mul_i.fix_variables(partial_point);
        let w_b = self.w_b.fix_variables(b_partial);
        let w_c = self.w_c.fix_variables(c_partial);

        Self {
            add_i,
            mul_i,
            w_b,
            w_c,
        }
    }

    fn to_univariate(&self) -> univariate::SparsePolynomial<F> {
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
        self.add_i.num_vars()
    }

    fn to_evaluations(&self) -> Vec<F> {
        // combine the evaluations of separate multilinear
        // extensions into a vector of evaluations of the
        // whole polynomial
        let w_b_evals = self.w_b.to_evaluations();
        let w_c_evals = self.w_c.to_evaluations();
        let add_i_evals = self.add_i.to_evaluations();
        let mul_i_evals = self.mul_i.to_evaluations();

        let mut res = vec![];
        for (b_idx, w_b_item) in w_b_evals.iter().enumerate() {
            for (c_idx, w_c_item) in w_c_evals.iter().enumerate() {
                let bc_idx = idx(c_idx, b_idx, self.w_b.num_vars());

                res.push(
                    add_i_evals[bc_idx] * (*w_b_item + w_c_item)
                        + mul_i_evals[bc_idx] * (*w_b_item * w_c_item),
                );
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
