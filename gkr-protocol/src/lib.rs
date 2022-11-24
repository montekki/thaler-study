use std::{
    cmp, iter,
    ops::{Add, Mul},
};

use ark_ff::{FftField, Field, Zero};
use ark_poly::{
    univariate, DenseMultilinearExtension, DenseUVPolynomial, EvaluationDomain, Evaluations,
    GeneralEvaluationDomain, MultilinearExtension, Polynomial,
};
use ark_std::rand::Rng;

use sum_check_protocol::{
    Prover as SumCheckProver, SumCheckPolynomial, Verifier as SumCheckVerifier,
    VerifierRoundResult as SumCheckVerifierRoundResult,
};

#[derive(Debug)]
pub enum Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// A type of a gate in the Circuit.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum GateType {
    /// An addition gate.
    Add,

    /// A multiplication gate.
    Mul,
}

/// A gate in the Circuit.
#[derive(Clone, Copy)]
pub struct Gate {
    /// A type of the gate.
    ttype: GateType,

    /// Two inputs, indexes into the previous layer gates outputs.
    inputs: [usize; 2],
}

impl Gate {
    /// Create a new `Gate`.
    pub fn new(ttype: GateType, inputs: [usize; 2]) -> Self {
        Self { ttype, inputs }
    }
}

/// A layer of gates in the circuit.
#[derive(Clone)]
pub struct CircuitLayer {
    layer: Vec<Gate>,
}

impl CircuitLayer {
    /// Create a new `CircuitLayer`.
    pub fn new(layer: Vec<Gate>) -> Self {
        Self { layer }
    }

    /// The length of the layer.
    pub fn len(&self) -> usize {
        self.layer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layer.is_empty()
    }
}

/// An evaluation of a `Circuit` on some input.
/// Stores every circuit layer interediary evaluations and the
/// circuit evaluation outputs.
pub struct CircuitEvaluation<F> {
    /// Evaluations on per-layer basis.
    pub layers: Vec<Vec<F>>,
}

impl<F: Copy> CircuitEvaluation<F> {
    /// Takes a gate label and outputs the corresponding gate's value at layer `layer`.
    pub fn w(&self, layer: usize, label: usize) -> F {
        self.layers[layer][label]
    }
}

/// The circuit in layered form.
#[derive(Clone)]
pub struct Circuit {
    /// First layer being the output layer, last layer being
    /// the input layer.
    layers: Vec<CircuitLayer>,

    /// Number of inputs
    num_inputs: usize,
}

impl Circuit {
    /// Evaluate a `Circuit` on a given input.
    pub fn evaluate<F>(&self, input: &[F]) -> CircuitEvaluation<F>
    where
        F: Add<Output = F> + Mul<Output = F> + Copy,
    {
        let mut layers = vec![];
        let mut current_input = input;

        layers.push(input.to_vec());

        for layer in self.layers.iter().rev() {
            let temp_layer: Vec<_> = layer
                .layer
                .iter()
                .map(|e| match e.ttype {
                    GateType::Add => current_input[e.inputs[0]] + current_input[e.inputs[1]],
                    GateType::Mul => current_input[e.inputs[0]] * current_input[e.inputs[1]],
                })
                .collect();

            layers.push(temp_layer);
            current_input = &layers[layers.len() - 1];
        }

        layers.reverse();
        CircuitEvaluation { layers }
    }

    /// The $\text{add}_i(a, b, c)$ polynomial value at layer $i$.
    pub fn add_i(&self, i: usize, a: usize, b: usize, c: usize) -> bool {
        let gate = &self.layers[i].layer[a];

        gate.ttype == GateType::Add && gate.inputs[0] == b && gate.inputs[1] == c
    }

    /// The $\text{mul}_i(a, b, c)$ polynomial value at layer $i$.
    pub fn mul_i(&self, i: usize, a: usize, b: usize, c: usize) -> bool {
        let gate = &self.layers[i].layer[a];

        gate.ttype == GateType::Mul && gate.inputs[0] == b && gate.inputs[1] == c
    }

    pub fn num_outputs(&self) -> usize {
        self.layers[0].layer.len()
    }

    fn add_i_ext<F: Field>(&self, r_i: &[F], i: usize) -> DenseMultilinearExtension<F> {
        let mut add_i = vec![];
        let num_vars_current = f64::from(self.layers[i].len() as u32).log2() as usize;

        let num_vars_next = f64::from(
            self.layers
                .get(i + 1)
                .map(|c| c.len())
                .unwrap_or(self.num_inputs) as u32,
        )
        .log2() as usize;

        for c in 0..2usize.pow(num_vars_next as u32) {
            for b in 0..2usize.pow(num_vars_next as u32) {
                for a in 0..2usize.pow(num_vars_current as u32) {
                    add_i.push(match self.add_i(i, a, b, c) {
                        true => F::one(),
                        false => F::zero(),
                    });
                }
            }
        }

        let add_i = DenseMultilinearExtension::from_evaluations_vec(
            num_vars_current + num_vars_next * 2,
            add_i,
        );

        add_i.fix_variables(r_i)
    }

    fn mul_i_ext<F: Field>(&self, r_i: &[F], i: usize) -> DenseMultilinearExtension<F> {
        let mut mul_i = vec![];
        let num_vars_current = f64::from(self.layers[i].len() as u32).log2() as usize;

        let num_vars_next = f64::from(
            self.layers
                .get(i + 1)
                .map(|c| c.len())
                .unwrap_or(self.num_inputs) as u32,
        )
        .log2() as usize;

        for c in 0..2usize.pow(num_vars_next as u32) {
            for b in 0..2usize.pow(num_vars_next as u32) {
                for a in 0..2usize.pow(num_vars_current as u32) {
                    mul_i.push(match self.mul_i(i, a, b, c) {
                        true => F::one(),
                        false => F::zero(),
                    });
                }
            }
        }

        let mul_i = DenseMultilinearExtension::from_evaluations_vec(
            num_vars_current + num_vars_next * 2,
            mul_i,
        );

        mul_i.fix_variables(r_i)
    }
}

/// A $2k_{i+1}$ variate polynomial used for each step of GKR protocol.
///
/// $$
/// f^{i}_{r_i}(b, c) \coloneqq
/// \widetilde{add}_i(r_i, b, c)(\tilde{W}\_{i+1}(b) + \tilde{W}\_{i+1}(c)) +
/// \widetilde{mul}_i(r_i, b, c)(\tilde{W}\_{i+1}(b) \cdot \tilde{W}\_{i+1}(c))
/// $$
#[derive(Clone)]
pub struct W<F: Field> {
    add_i: DenseMultilinearExtension<F>,
    mul_i: DenseMultilinearExtension<F>,
    w_b: DenseMultilinearExtension<F>,
    w_c: DenseMultilinearExtension<F>,
}

impl<F: Field> W<F> {
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
        let (b, c) = point.split_at(self.w_b.num_vars());
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

    fn to_univariate(&self) -> ark_poly::univariate::SparsePolynomial<F> {
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

/// The state of the Verifier.
pub struct Verifier<F: FftField> {
    /// $r_0, r_1,..., r_n$.
    r: Vec<Vec<F>>,

    /// $m_0$.
    m: Vec<F>,

    /// $b$ and $c$.
    bc: Vec<F>,

    /// Sum Check Verifier for the current circuit layer
    verifier: Option<SumCheckVerifier<F, W<F>>>,

    /// Circuit
    circuit: Circuit,

    /// %add_i$
    add_i: DenseMultilinearExtension<F>,

    /// $mul_i$
    mul_i: DenseMultilinearExtension<F>,

    /// Current layer.
    i: usize,
}

impl<F: FftField> Verifier<F> {
    /// Create a new `Verifier` with the claim of the `Prover`.
    ///
    /// At the start of the protocol picks a random
    /// $r_0 \in \mathbb{F}^{k_0}$
    /// and sets
    /// $m_0 \leftarrow \tilde{D}(r_0)$.
    /// The remainder of the protocol is devoted to confirming
    /// that
    /// $m_0 = \tilde{W}_0(r_0)$
    pub fn new<R: Rng>(num_outputs: usize, d: &[F], circuit: Circuit, rng: &mut R) -> Self {
        let num_vars = f64::from(num_outputs as u32).log2() as usize;
        let d = DenseMultilinearExtension::from_evaluations_slice(num_vars, d);

        let r_zero: Vec<_> = (0..num_vars).map(|_| F::rand(rng)).collect();

        let m_zero = d.evaluate(&r_zero).unwrap();

        let add_i = circuit.add_i_ext(&r_zero, 0);

        let mul_i = circuit.mul_i_ext(&r_zero, 0);

        let r = vec![r_zero];

        let m = vec![m_zero];
        Self {
            r,
            m,
            bc: vec![],
            verifier: None,
            circuit,
            i: 0,
            add_i,
            mul_i,
        }
    }

    pub fn start_round(&mut self, c_1: F, round: usize, num_vars: usize) {
        self.add_i = self.circuit.add_i_ext(self.r.last().unwrap(), round);

        self.mul_i = self.circuit.mul_i_ext(self.r.last().unwrap(), round);
        self.i = round;
        self.verifier = Some(SumCheckVerifier::new(num_vars, c_1, None));
        self.bc = vec![];
    }

    pub fn r(&self, i: usize) -> Vec<F> {
        self.r[i].clone()
    }

    pub fn final_random_point<R: Rng>(&mut self, rng: &mut R) -> F {
        let final_point = F::rand(rng);
        self.bc.push(final_point);
        final_point
    }

    pub fn receive_prover_msg<R: Rng>(
        &mut self,
        msg: ProverMessage<F>,
        rng: &mut R,
    ) -> Result<VerifierMessage<F>> {
        match msg {
            ProverMessage::SumCheckProverMessage { p } => {
                let res = self.verifier.as_mut().unwrap().round(p, rng).unwrap();

                if let SumCheckVerifierRoundResult::JthRound(point) = res {
                    self.bc.push(point);
                }

                Ok(VerifierMessage::SumCheckRoundResult { res })
            }
            ProverMessage::FinalRoundMessage { p, q } => {
                /*
                 * TODO: check q degree
                 */

                let q_0 = q.evaluate(&F::zero());
                let q_1 = q.evaluate(&F::one());

                let eval = self.add_i.evaluate(&self.bc).unwrap() * (q_0 + q_1)
                    + self.mul_i.evaluate(&self.bc).unwrap() * q_0 * q_1;

                assert_eq!(eval, p.evaluate(self.bc.last().unwrap()));

                let r = F::rand(rng);
                let (b, c) = self.bc.split_at(self.bc.len() / 2);

                let line = line(b, c);

                let r_next: Vec<F> = line.into_iter().map(|e| e.evaluate(&r)).collect();
                let m_next = q.evaluate(&r);

                self.r.push(r_next);
                self.m.push(m_next);

                Ok(VerifierMessage::LastRoundResult)
            }
        }
    }

    pub fn check_input(&self, input: &[F]) -> bool {
        let w = DenseMultilinearExtension::from_evaluations_slice(
            (f64::from(input.len() as u32)).log2() as usize,
            input,
        );

        &w.evaluate(self.r.last().unwrap()).unwrap() == self.m.last().unwrap()
    }
}

#[derive(Debug)]
pub enum VerifierMessage<F: Field> {
    SumCheckRoundResult { res: SumCheckVerifierRoundResult<F> },
    LastRoundResult,
}

#[derive(Debug)]
pub enum ProverMessage<F: Field> {
    SumCheckProverMessage {
        p: univariate::SparsePolynomial<F>,
    },
    FinalRoundMessage {
        p: univariate::SparsePolynomial<F>,

        /// Sends a univariate polynomial $q$ of degree at most
        /// k_{i+1} claimed to equal $\tilde{W}_{i+1}$ to $l$
        q: univariate::SparsePolynomial<F>,
    },
}

fn line<F: Field>(b: &[F], c: &[F]) -> Vec<univariate::SparsePolynomial<F>> {
    iter::zip(b, c)
        .map(|(b, c)| {
            univariate::SparsePolynomial::from_coefficients_slice(&[(0, *b), (1, *c - b)])
        })
        .collect()
}

fn restrict_poly<F: Field, M: MultilinearExtension<F>>(
    b: &[F],
    c: &[F],
    mle: &M,
) -> univariate::SparsePolynomial<F> {
    let k: Vec<_> = iter::zip(b, c).map(|(b, c)| *c - b).collect();

    let evaluations = mle.to_evaluations();
    let num_vars = mle.num_vars();

    let mut res = univariate::SparsePolynomial::zero();

    for (i, evaluation) in evaluations.iter().enumerate() {
        let mut p = univariate::SparsePolynomial::from_coefficients_vec(vec![(0, *evaluation)]);
        for bit in 0..num_vars {
            let mut b =
                univariate::SparsePolynomial::from_coefficients_vec(vec![(0, b[bit]), (1, k[bit])]);

            if i & (1 << bit) == 0 {
                b = (&univariate::DensePolynomial::from_coefficients_vec(vec![F::one()]) - &b)
                    .into();
            }

            p = p.mul(&b);
        }

        res += &p;
    }

    res
}

/// The state of the Prover.
pub struct Prover<F: FftField> {
    i: usize,
    circuit: Circuit,
    evaluation: CircuitEvaluation<F>,
    prover: Option<SumCheckProver<F, W<F>>>,
    w: DenseMultilinearExtension<F>,
    r: Vec<F>,
}

impl<F: FftField> Prover<F> {
    /// Create a new `Prover` state from a circuit and an evaluation.
    pub fn new(circuit: Circuit, input: &[F]) -> Self {
        let evaluation = circuit.evaluate(input);

        Self {
            i: 0,
            circuit,
            evaluation,
            prover: None,
            w: Default::default(),
            r: vec![],
        }
    }

    /// At the start of the protocol $P$ sends a function
    /// $D: \lbrace 0, 1 \rbrace ^{k_0} \rightarrow \mathbb{F}$
    /// claimed to equal $W_0$ (the function mapping output gate
    /// labels to output values).
    pub fn start_protocol(&self) -> Vec<F> {
        self.evaluation.layers.first().unwrap().clone()
    }

    /// Create a Sum-Check prover for round $i$.
    ///
    /// At round $i$ a Sum-Check prover for polynomial
    /// $f^{(i)}_{r_i}(b, c)$.
    pub fn start_round(&mut self, i: usize, r_i: &[F]) {
        let num_vars_current = f64::from(self.evaluation.layers[i].len() as u32).log2() as usize;

        let num_vars_next = f64::from(self.evaluation.layers[i + 1].len() as u32).log2() as usize;
        let w_b = DenseMultilinearExtension::from_evaluations_slice(
            num_vars_next,
            &self.evaluation.layers[i + 1],
        );
        self.w = w_b.clone();
        let w_c = w_b.clone();
        let mut add_i = vec![];
        let mut mult_i = vec![];

        for c in 0..2usize.pow(num_vars_next as u32) {
            for b in 0..2usize.pow(num_vars_next as u32) {
                for a in 0..2usize.pow(num_vars_current as u32) {
                    add_i.push(match self.circuit.add_i(i, a, b, c) {
                        true => F::one(),
                        false => F::zero(),
                    });

                    mult_i.push(match self.circuit.mul_i(i, a, b, c) {
                        true => F::one(),
                        false => F::zero(),
                    });
                }
            }
        }

        let add_i = DenseMultilinearExtension::from_evaluations_vec(
            num_vars_current + num_vars_next * 2,
            add_i,
        );
        let mult_i = DenseMultilinearExtension::from_evaluations_vec(
            num_vars_current + num_vars_next * 2,
            mult_i,
        );

        let add_i = add_i.fix_variables(r_i);
        let mult_i = mult_i.fix_variables(r_i);

        assert_eq!(add_i.num_vars(), mult_i.num_vars());
        assert_eq!(add_i.num_vars(), 2 * w_b.num_vars());

        let w = W::new(add_i, mult_i, w_b, w_c);
        self.i = i;
        self.prover = Some(SumCheckProver::new(w));
        self.r = vec![];
    }

    pub fn round_msg(&mut self, j: usize) -> ProverMessage<F> {
        if j == 2 * f64::from(
            self.circuit
                .layers
                .get(self.i + 1)
                .map(|c| c.len())
                .unwrap_or(self.circuit.num_inputs) as u32,
        )
        .log2() as usize
            - 1
        {
            // The last round; do the polynomial restriction.
            let (b, c) = self.r.split_at(self.r.len() / 2);

            let q = restrict_poly(b, c, &self.w);

            let p = self
                .prover
                .as_mut()
                .unwrap()
                .round(self.r[j - 1], j)
                .unwrap();
            ProverMessage::FinalRoundMessage { p, q }
        } else {
            // Just a Sum-Check round
            let point = if j == 0 { F::one() } else { self.r[j - 1] };

            ProverMessage::SumCheckProverMessage {
                p: self.prover.as_mut().unwrap().round(point, j).unwrap(),
            }
        }
    }

    pub fn receive_verifier_msg(&mut self, verifier_msg: VerifierMessage<F>) {
        match verifier_msg {
            VerifierMessage::SumCheckRoundResult { res } => match res {
                SumCheckVerifierRoundResult::JthRound(r_j) => {
                    self.r.push(r_j);
                }
                SumCheckVerifierRoundResult::FinalRound(_) => panic!(),
            },
            VerifierMessage::LastRoundResult => (),
        }
    }

    pub fn c_1(&self) -> F {
        self.prover.as_ref().unwrap().c_1()
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{Fp64, MontBackend, MontConfig, PrimeField};
    use ark_poly::univariate::DensePolynomial;
    use ark_std::test_rng;
    use pretty_assertions::assert_eq;

    use super::*;

    fn circuit_from_book() -> Circuit {
        Circuit {
            layers: vec![
                CircuitLayer {
                    layer: vec![
                        Gate {
                            ttype: GateType::Mul,
                            inputs: [0, 1],
                        },
                        Gate {
                            ttype: GateType::Mul,
                            inputs: [2, 3],
                        },
                    ],
                },
                CircuitLayer {
                    layer: vec![
                        Gate {
                            ttype: GateType::Mul,
                            inputs: [0, 0],
                        },
                        Gate {
                            ttype: GateType::Mul,
                            inputs: [1, 1],
                        },
                        Gate {
                            ttype: GateType::Mul,
                            inputs: [1, 2],
                        },
                        Gate {
                            ttype: GateType::Mul,
                            inputs: [3, 3],
                        },
                    ],
                },
            ],
            num_inputs: 4,
        }
    }

    fn three_layer_circuit() -> Circuit {
        Circuit {
            layers: vec![
                CircuitLayer {
                    layer: vec![
                        Gate {
                            ttype: GateType::Add,
                            inputs: [0, 1],
                        },
                        Gate {
                            ttype: GateType::Add,
                            inputs: [2, 3],
                        },
                    ],
                },
                CircuitLayer {
                    layer: vec![
                        Gate {
                            ttype: GateType::Add,
                            inputs: [0, 1],
                        },
                        Gate {
                            ttype: GateType::Add,
                            inputs: [2, 3],
                        },
                        Gate {
                            ttype: GateType::Add,
                            inputs: [4, 5],
                        },
                        Gate {
                            ttype: GateType::Add,
                            inputs: [6, 7],
                        },
                    ],
                },
            ],
            num_inputs: 8,
        }
    }

    #[test]
    /// Test restrict poly
    fn test_restrict_poly() {
        #[derive(MontConfig)]
        #[modulus = "389"]
        #[generator = "2"]
        struct FrConfig;

        type Fp389 = Fp64<MontBackend<FrConfig, 1>>;

        let b = [
            Fp389::from_bigint(2u32.into()).unwrap(),
            Fp389::from_bigint(4u32.into()).unwrap(),
        ];

        let c = [
            Fp389::from_bigint(3u32.into()).unwrap(),
            Fp389::from_bigint(2u32.into()).unwrap(),
        ];

        let evaluations = [
            Fp389::from_bigint(0u32.into()).unwrap(),
            Fp389::from_bigint(0u32.into()).unwrap(),
            Fp389::from_bigint(2u32.into()).unwrap(),
            Fp389::from_bigint(5u32.into()).unwrap(),
        ];

        let poly = restrict_poly(
            &b,
            &c,
            &DenseMultilinearExtension::from_evaluations_slice(2, &evaluations),
        );
        let dense: DensePolynomial<Fp389> = poly.into();
        // -6t^2 - 4t + 32
        assert_eq!(
            vec![32, 385, 383],
            dense
                .coeffs()
                .iter()
                .map(|c| c.into_bigint().as_ref()[0])
                .collect::<Vec<_>>()
        );
    }

    /// A test of the circuit from figure 4.12
    #[test]
    fn circuit_test_from_book() {
        let circuit = circuit_from_book();

        let layers = circuit.evaluate(&[3, 2, 3, 1]);
        assert_eq!(
            layers.layers,
            vec![vec![36, 6], vec![9, 4, 6, 1], vec![3, 2, 3, 1]]
        );

        // Test that mul_1 evaluates to 0 on all inputs except
        // ((0, 0), (0, 0), (0, 0))
        // ((0, 1), (0, 1), (0, 1))
        // ((1, 0), (0, 1), (1, 0))
        // ((1, 1), (1, 1), (1, 1))
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    let expected = ((a == 0 || a == 1) && a == b && a == c)
                        || a == 2 && b == 1 && c == 2
                        || a == b && b == c && a == 3;
                    assert_eq!(circuit.mul_i(1, a, b, c), expected, "{a} {b} {c}");
                }
            }
        }
    }

    #[test]
    fn protocol_test_from_book() {
        let rng = &mut test_rng();
        #[derive(MontConfig)]
        #[modulus = "389"]
        #[generator = "2"]
        struct FrConfig;

        type Fp389 = Fp64<MontBackend<FrConfig, 1>>;

        let circuit = circuit_from_book();

        let input = [
            Fp389::from_bigint(3u32.into()).unwrap(),
            Fp389::from_bigint(2u32.into()).unwrap(),
            Fp389::from_bigint(3u32.into()).unwrap(),
            Fp389::from_bigint(1u32.into()).unwrap(),
        ];

        let expected_outputs = [
            Fp389::from_bigint(36u32.into()).unwrap(),
            Fp389::from_bigint(6u32.into()).unwrap(),
        ];

        let num_outputs = circuit.num_outputs();

        let mut prover = Prover::new(circuit.clone(), &input);

        // At the start of the protocol Prover sends a function $W_0$
        // mapping output gate labels to output values.
        let circuit_outputs = prover.start_protocol();

        assert_eq!(circuit_outputs, expected_outputs);

        let mut verifier = Verifier::new(num_outputs, &circuit_outputs, circuit.clone(), rng);

        for i in 0..circuit.layers.len() {
            let r_i = verifier.r(i);
            prover.start_round(i, &r_i);

            let c_1 = prover.c_1();
            let num_vars = 2 * f64::from(
                circuit
                    .layers
                    .get(i + 1)
                    .map(|c| c.len())
                    .unwrap_or(circuit.num_inputs) as u32,
            )
            .log2() as usize;

            verifier.start_round(c_1, i, num_vars);

            for j in 0..(num_vars - 1) {
                let prover_msg = prover.round_msg(j);

                let verifier_msg = verifier.receive_prover_msg(prover_msg, rng).unwrap();

                prover.receive_verifier_msg(verifier_msg);
            }

            let last_rand = verifier.final_random_point(rng);
            prover.receive_verifier_msg(VerifierMessage::SumCheckRoundResult {
                res: SumCheckVerifierRoundResult::JthRound(last_rand),
            });

            let prover_msg = prover.round_msg(num_vars - 1);
            let verifier_msg = verifier.receive_prover_msg(prover_msg, rng).unwrap();
            match verifier_msg {
                VerifierMessage::LastRoundResult => (),
                _ => panic!("{:?}", verifier_msg),
            }
        }

        assert!(verifier.check_input(&input));
    }

    #[test]
    fn three_layer_protocol_test() {
        let rng = &mut test_rng();
        #[derive(MontConfig)]
        #[modulus = "389"]
        #[generator = "2"]
        struct FrConfig;

        type Fp389 = Fp64<MontBackend<FrConfig, 1>>;

        let circuit = three_layer_circuit();

        let input = [
            Fp389::from_bigint(0u32.into()).unwrap(),
            Fp389::from_bigint(1u32.into()).unwrap(),
            Fp389::from_bigint(0u32.into()).unwrap(),
            Fp389::from_bigint(1u32.into()).unwrap(),
            Fp389::from_bigint(0u32.into()).unwrap(),
            Fp389::from_bigint(1u32.into()).unwrap(),
            Fp389::from_bigint(0u32.into()).unwrap(),
            Fp389::from_bigint(1u32.into()).unwrap(),
        ];

        let expected_outputs = [
            Fp389::from_bigint(2u32.into()).unwrap(),
            Fp389::from_bigint(2u32.into()).unwrap(),
        ];

        let mut prover = Prover::new(circuit.clone(), &input);

        let circuit_outputs = prover.start_protocol();

        assert_eq!(circuit_outputs, expected_outputs);

        let mut verifier = Verifier::new(
            circuit.num_outputs(),
            &circuit_outputs,
            circuit.clone(),
            rng,
        );

        for i in 0..circuit.layers.len() {
            let r_i = verifier.r(i);
            prover.start_round(i, &r_i);

            let c_1 = prover.c_1();
            let num_vars = 2 * f64::from(
                circuit
                    .layers
                    .get(i + 1)
                    .map(|c| c.len())
                    .unwrap_or(circuit.num_inputs) as u32,
            )
            .log2() as usize;

            verifier.start_round(c_1, i, num_vars);

            for j in 0..(num_vars - 1) {
                let prover_msg = prover.round_msg(j);

                let verifier_msg = verifier.receive_prover_msg(prover_msg, rng).unwrap();

                prover.receive_verifier_msg(verifier_msg);
            }

            let last_rand = verifier.final_random_point(rng);
            prover.receive_verifier_msg(VerifierMessage::SumCheckRoundResult {
                res: SumCheckVerifierRoundResult::JthRound(last_rand),
            });

            let prover_msg = prover.round_msg(num_vars - 1);
            let verifier_msg = verifier.receive_prover_msg(prover_msg, rng).unwrap();

            match verifier_msg {
                VerifierMessage::LastRoundResult => (),
                _ => panic!("{:?}", verifier_msg),
            }
        }

        assert!(verifier.check_input(&input));
    }
}
