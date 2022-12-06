#[deny(missing_docs)]
use std::{cmp, iter};

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

mod circuit;

use circuit::{Circuit, CircuitEvaluation};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Verifier is in the wrong state.")]
    WrongVerifierState,
}

pub type Result<T> = std::result::Result<T, Error>;

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

/// The state of the Verifier.
pub struct Verifier<F: FftField> {
    /// $r_0, r_1,..., r_n$.
    r: Vec<Vec<F>>,

    /// $m$.
    m: Vec<F>,

    /// Circuit
    circuit: Circuit,

    /// State of the verifier.
    state: VerifierState<F>,
}

/// The inner state of the [`Verifier`].
enum VerifierState<F: FftField> {
    Empty,
    RunningSumCheck {
        /// $b$ and $c$.
        bc: Vec<F>,

        /// Verifier.
        verifier: Box<SumCheckVerifier<F, W<F>>>,

        /// $add_i$
        add_i: DenseMultilinearExtension<F>,

        /// $mul_i$
        mul_i: DenseMultilinearExtension<F>,
    },
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
    pub fn new(circuit: Circuit) -> Self {
        Self {
            r: vec![],
            m: vec![],
            circuit,
            state: VerifierState::Empty,
        }
    }

    fn start_round(&mut self, c_1: F, round: usize, num_vars: usize) -> Result<VerifierMessage<F>> {
        let add_i = self.circuit.add_i_ext(self.r.last().unwrap(), round);
        let mul_i = self.circuit.mul_i_ext(self.r.last().unwrap(), round);
        let verifier = SumCheckVerifier::new(num_vars, c_1, None);
        let bc = vec![];

        self.state = VerifierState::RunningSumCheck {
            bc,
            verifier: Box::new(verifier),
            add_i,
            mul_i,
        };

        Ok(VerifierMessage::RoundStarted(round))
    }

    pub fn r(&self, i: usize) -> Vec<F> {
        self.r[i].clone()
    }

    pub fn final_random_point<R: Rng>(&mut self, rng: &mut R) -> Result<F> {
        if let VerifierState::RunningSumCheck { bc, .. } = &mut self.state {
            let final_point = F::rand(rng);
            bc.push(final_point);

            Ok(final_point)
        } else {
            Err(Error::WrongVerifierState)
        }
    }

    fn sum_check_step<R: Rng>(
        &mut self,
        message: univariate::SparsePolynomial<F>,
        rng: &mut R,
    ) -> Result<VerifierMessage<F>> {
        if let VerifierState::RunningSumCheck { bc, verifier, .. } = &mut self.state {
            let res = verifier.round(message, rng).unwrap();

            if let SumCheckVerifierRoundResult::JthRound(point) = res {
                bc.push(point);
            }

            Ok(VerifierMessage::SumCheckRoundResult { res })
        } else {
            Err(Error::WrongVerifierState)
        }
    }

    fn final_round_message<R: Rng>(
        &mut self,
        p: univariate::SparsePolynomial<F>,
        q: univariate::SparsePolynomial<F>,
        rng: &mut R,
    ) -> Result<VerifierMessage<F>> {
        if let VerifierState::RunningSumCheck {
            bc, add_i, mul_i, ..
        } = &self.state
        {
            /*
             * TODO: check q degree
             */
            let q_0 = q.evaluate(&F::zero());
            let q_1 = q.evaluate(&F::one());

            let eval =
                add_i.evaluate(bc).unwrap() * (q_0 + q_1) + mul_i.evaluate(bc).unwrap() * q_0 * q_1;

            assert_eq!(eval, p.evaluate(bc.last().unwrap()));

            let r = F::rand(rng);
            let (b, c) = bc.split_at(bc.len() / 2);

            let line = line(b, c);

            let r_next: Vec<F> = line.into_iter().map(|e| e.evaluate(&r)).collect();
            let m_next = q.evaluate(&r);

            self.r.push(r_next);
            self.m.push(m_next);

            Ok(VerifierMessage::LastRoundResult)
        } else {
            Err(Error::WrongVerifierState)
        }
    }

    pub fn receive_prover_msg<R: Rng>(
        &mut self,
        msg: ProverMessage<F>,
        rng: &mut R,
    ) -> Result<VerifierMessage<F>> {
        match msg {
            ProverMessage::SumCheckProverMessage { p } => self.sum_check_step(p, rng),
            ProverMessage::StartSumCheck {
                c_1,
                round,
                num_vars,
            } => self.start_round(c_1, round, num_vars),
            ProverMessage::FinalRoundMessage { p, q } => self.final_round_message(p, q, rng),
            ProverMessage::Begin { circuit_outputs } => {
                let num_output_vars = self.circuit.num_vars_at(0).unwrap();
                let d = DenseMultilinearExtension::from_evaluations_slice(
                    num_output_vars,
                    &circuit_outputs,
                );

                let r_zero: Vec<_> = (0..num_output_vars).map(|_| F::rand(rng)).collect();

                let m_zero = d.evaluate(&r_zero).unwrap();

                self.r = vec![r_zero];
                self.m = vec![m_zero];

                Ok(VerifierMessage::FirstRound)
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

/// Messages emitted by the [`Verifier`].
#[derive(Debug)]
pub enum VerifierMessage<F: Field> {
    /// A result of running a step in the current sum check protocol.
    SumCheckRoundResult { res: SumCheckVerifierRoundResult<F> },
    /// The last round has completed.
    LastRoundResult,
    /// The first round has completed.
    FirstRound,
    /// The j-th round has started.
    RoundStarted(usize),
}

/// Messages emitted by the [`Prover`].
#[derive(Debug, PartialEq, Eq)]
pub enum ProverMessage<F: Field> {
    /// [`Prover`] begins the protocol by the claim about the outputs.
    Begin { circuit_outputs: Vec<F> },
    /// A step of the current sum-check protocol.
    SumCheckProverMessage { p: univariate::SparsePolynomial<F> },
    /// In the final the restriction polynomial $q$ is added.
    FinalRoundMessage {
        p: univariate::SparsePolynomial<F>,

        /// Sends a univariate polynomial $q$ of degree at most
        /// k_{i+1} claimed to equal $\tilde{W}_{i+1}$ to $l$
        q: univariate::SparsePolynomial<F>,
    },
    /// Instruct the [`Verifier`] to start a Sum-Check protocol for some round.
    StartSumCheck {
        c_1: F,
        round: usize,
        num_vars: usize,
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
    /// Current round of the protocol.
    i: usize,

    /// The circuit.
    circuit: Circuit,

    /// Evaluations of the circuit on a given input.
    evaluation: CircuitEvaluation<F>,

    /// A Sum-Check protocol prover.
    prover: Option<SumCheckProver<F, W<F>>>,

    /// Current $\tilde{W}_{i+1}$.
    w: DenseMultilinearExtension<F>,

    /// Random points collected through a single Sum-Check protocol run.
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
    pub fn start_protocol(&self) -> ProverMessage<F> {
        ProverMessage::Begin {
            circuit_outputs: self.evaluation.layers.first().unwrap().clone(),
        }
    }

    /// Create a Sum-Check prover for round $i$.
    ///
    /// At round $i$ a Sum-Check prover for polynomial
    /// $f^{(i)}_{r_i}(b, c)$.
    pub fn start_round(&mut self, i: usize, r_i: &[F]) -> ProverMessage<F> {
        let num_vars_current = self.circuit.num_vars_at(i).unwrap();

        let num_vars_next = self.circuit.num_vars_at(i + 1).unwrap();

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

        let num_vars = add_i.num_vars();

        assert_eq!(add_i.num_vars(), mult_i.num_vars());
        assert_eq!(add_i.num_vars(), 2 * w_b.num_vars());

        let w = W::new(add_i, mult_i, w_b, w_c);
        self.i = i;

        let prover = SumCheckProver::new(w);
        let c_1 = prover.c_1();
        self.prover = Some(prover);
        self.r = vec![];

        ProverMessage::StartSumCheck {
            c_1,
            round: i,
            num_vars,
        }
    }

    /// Perform a step of the Sum-Check protocol and provide a message for the [`Verifier`].
    pub fn round_msg(&mut self, j: usize) -> ProverMessage<F> {
        if j == 2 * self.circuit.num_vars_at(self.i + 1).unwrap() - 1 {
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

    /// Receive a message from the [`Verifier`].
    pub fn receive_verifier_msg(&mut self, verifier_msg: VerifierMessage<F>) {
        match verifier_msg {
            VerifierMessage::SumCheckRoundResult { res } => match res {
                SumCheckVerifierRoundResult::JthRound(r_j) => {
                    self.r.push(r_j);
                }
                SumCheckVerifierRoundResult::FinalRound(_) => panic!(),
            },
            VerifierMessage::LastRoundResult => panic!(),
            _ => (),
        }
    }

    /// Get the $c_1$ of the current Sum-Check prover.
    pub fn c_1(&self) -> F {
        self.prover.as_ref().unwrap().c_1()
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{Fp64, MontBackend, MontConfig, PrimeField};
    use ark_poly::univariate::DensePolynomial;
    use ark_std::test_rng;
    use circuit::circuit_from_book;
    use pretty_assertions::assert_eq;

    use crate::circuit::{CircuitLayer, Gate, GateType};

    use super::*;

    fn three_layer_circuit() -> Circuit {
        Circuit::new(
            vec![
                CircuitLayer::new(vec![
                    Gate::new(GateType::Add, [0, 1]),
                    Gate::new(GateType::Add, [2, 3]),
                ]),
                CircuitLayer::new(vec![
                    Gate::new(GateType::Add, [0, 1]),
                    Gate::new(GateType::Add, [2, 3]),
                    Gate::new(GateType::Add, [4, 5]),
                    Gate::new(GateType::Add, [6, 7]),
                ]),
            ],
            8,
        )
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

        let mut prover = Prover::new(circuit.clone(), &input);

        // At the start of the protocol Prover sends a function $W_0$
        // mapping output gate labels to output values.
        let circuit_outputs_message = prover.start_protocol();

        assert_eq!(
            circuit_outputs_message,
            ProverMessage::Begin {
                circuit_outputs: expected_outputs.to_vec()
            }
        );

        let mut verifier = Verifier::new(circuit.clone());
        verifier
            .receive_prover_msg(circuit_outputs_message, rng)
            .unwrap();

        for i in 0..circuit.layers().len() {
            let r_i = verifier.r(i);
            let msg = prover.start_round(i, &r_i);

            let num_vars = 2 * circuit.num_vars_at(i + 1).unwrap();

            verifier.receive_prover_msg(msg, rng).unwrap();

            for j in 0..(num_vars - 1) {
                let prover_msg = prover.round_msg(j);

                let verifier_msg = verifier.receive_prover_msg(prover_msg, rng).unwrap();

                prover.receive_verifier_msg(verifier_msg);
            }

            let last_rand = verifier.final_random_point(rng).unwrap();
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

        let circuit_outputs_message = prover.start_protocol();

        assert_eq!(
            circuit_outputs_message,
            ProverMessage::Begin {
                circuit_outputs: expected_outputs.to_vec()
            }
        );

        let mut verifier = Verifier::new(circuit.clone());

        verifier
            .receive_prover_msg(circuit_outputs_message, rng)
            .unwrap();

        for i in 0..circuit.layers().len() {
            let r_i = verifier.r(i);
            let prover_msg = prover.start_round(i, &r_i);
            verifier.receive_prover_msg(prover_msg, rng).unwrap();
            let num_vars = 2 * circuit.num_vars_at(i + 1).unwrap();

            for j in 0..(num_vars - 1) {
                let prover_msg = prover.round_msg(j);

                let verifier_msg = verifier.receive_prover_msg(prover_msg, rng).unwrap();

                prover.receive_verifier_msg(verifier_msg);
            }

            let last_rand = verifier.final_random_point(rng).unwrap();
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