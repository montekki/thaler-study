#![deny(unused_crate_dependencies)]
#![deny(missing_docs)]

//! Fiat-Shamir Transformation implementation.

use ark_ff::{field_hashers::HashToField, Field};
use ark_poly::univariate;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use sum_check_protocol::{RngF, SumCheckPolynomial};

/// Crate error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error in ark_serialize.
    #[error("Codec error")]
    Serialization,

    /// A SumCheck error.
    #[error(transparent)]
    SumCheck(#[from] sum_check_protocol::Error),
}

impl From<SerializationError> for Error {
    fn from(_: SerializationError) -> Self {
        Self::Serialization
    }
}

/// Crate `Result` type.
pub type Result<T> = std::result::Result<T, Error>;

/// A trait describing an Interactive Prover.
pub trait InteractiveProver<F> {
    /// Get $g_1$.
    fn g_1(&mut self) -> Result<Vec<u8>>;

    /// Perform a step with V's challenge $r_i$.
    fn round(&mut self, j: usize, r_j: F) -> Result<Vec<u8>>;

    /// Number of rounds.
    fn num_rounds(&self) -> usize;
}

impl<F: Field, P: SumCheckPolynomial<F>> InteractiveProver<F> for sum_check_protocol::Prover<F, P> {
    fn g_1(&mut self) -> Result<Vec<u8>> {
        let mut res = vec![];

        let p: (F, univariate::SparsePolynomial<F>) = (self.c_1(), self.round(F::one(), 0));

        p.serialize_uncompressed(&mut res)?;

        Ok(res)
    }

    fn round(&mut self, j: usize, r_j: F) -> Result<Vec<u8>> {
        let mut res = vec![];

        sum_check_protocol::Prover::round(self, r_j, j).serialize_uncompressed(&mut res)?;

        Ok(res)
    }

    fn num_rounds(&self) -> usize {
        self.num_vars()
    }
}

/// A transcript of the Fiat-Shamir transformation.
pub struct FiatShamirTranscript {
    g: Vec<Vec<u8>>,
}

/// Generate a Fiat-Shamir transcript turning an Interactive Prover
/// into a non-interactive one.
pub fn generate_transcript<F: Field, P: InteractiveProver<F>, H: HashToField<F>>(
    mut prover: P,
) -> Result<FiatShamirTranscript> {
    let hasher = H::new(&[]);

    let g_1 = prover.g_1()?;

    let mut hash_input = vec![];
    hash_input.extend_from_slice(&g_1);

    let mut g = vec![g_1];

    for j in 1..prover.num_rounds() {
        let r_j = hasher.hash_to_field::<1>(&hash_input)[0];

        let g_j = prover.round(j, r_j)?;

        hash_input.extend_from_slice(&g_j);

        g.push(g_j);
    }

    Ok(FiatShamirTranscript { g })
}

/// A helper struct to feed non-random values into
/// interactive verifiers and provers.
pub struct RandNums<F> {
    nums: Vec<F>,
    current: usize,
}

impl<F: Field> RandNums<F> {
    fn new(nums: Vec<F>) -> Self {
        Self { nums, current: 0 }
    }
}

impl<F: Copy> RngF<F> for RandNums<F> {
    fn draw(&mut self) -> F {
        let res = self.nums[self.current];
        self.current += 1;
        res
    }
}

/// Perform verification of the Fiat-Shamir transcript
/// turning an Interactive Verifier into a non-interactive one.
pub fn verify_transcript<F: Field, V: InteractiveVerifier<F, RandNums<F>>, H: HashToField<F>>(
    transcript: FiatShamirTranscript,
    verifier: V,
) -> Result<bool> {
    let mut verifier = verifier;

    let hasher = H::new(&[]);

    let mut hash_input = vec![];

    for j in 0..transcript.g.len() {
        hash_input.extend_from_slice(&transcript.g[j]);
        let r_j = hasher.hash_to_field::<1>(&hash_input)[0];

        if !verifier.round(j, &transcript.g[j], &mut RandNums::new(vec![r_j]))? {
            return Ok(false);
        }
    }

    Ok(true)
}

/// A trait describing an Interactive Verifier.
pub trait InteractiveVerifier<F: Field, R: RngF<F>> {
    /// Perform a round of the Interactive Verifier.
    fn round(&mut self, j: usize, g_j: &[u8], rng: &mut R) -> Result<bool>;
}

impl<F: Field, R: RngF<F>, P: SumCheckPolynomial<F>> InteractiveVerifier<F, R>
    for sum_check_protocol::Verifier<F, P>
{
    fn round(&mut self, j: usize, g_j: &[u8], rng: &mut R) -> Result<bool> {
        if j == 0 {
            let c_1: (F, univariate::SparsePolynomial<F>) =
                CanonicalDeserialize::deserialize_uncompressed(g_j)?;
            self.set_c_1(c_1.0);
            self.round(c_1.1, rng)?;
            return Ok(true);
        }

        let g_j: univariate::SparsePolynomial<F> =
            CanonicalDeserialize::deserialize_uncompressed(g_j)?;

        match self.round(g_j, rng)? {
            sum_check_protocol::VerifierRoundResult::JthRound(_) => Ok(true),
            sum_check_protocol::VerifierRoundResult::FinalRound(res) => Ok(res),
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{field_hashers::DefaultFieldHasher, Field, Fp64, MontBackend, MontConfig};
    use ark_poly::{
        multivariate::{self, SparseTerm, Term},
        DenseMVPolynomial,
    };
    use ark_std::{rand::Rng, test_rng};
    use sum_check_protocol::{Prover, Verifier};

    use crate::{generate_transcript, verify_transcript};

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
                .filter_map(|i| {
                    if rng.gen_bool(0.5) {
                        Some((i, rng.gen_range(1..(d + 1))))
                    } else {
                        None
                    }
                })
                .collect();
            let coeff = F::rand(rng);
            random_terms.push((coeff, SparseTerm::new(term)));
        }
        multivariate::SparsePolynomial::from_coefficients_slice(l, &random_terms)
    }

    #[test]
    fn it_works() {
        use sha2::Sha256;
        let rng = &mut test_rng();
        for n in 2..10 {
            let g = rand_poly::<_, Fp5>(n, 3, rng);
            let prover = Prover::new(g.clone());
            let verifier = Verifier::new(n, Some(g));

            let transcript =
                generate_transcript::<_, _, DefaultFieldHasher<Sha256>>(prover).unwrap();

            assert!(
                verify_transcript::<_, _, DefaultFieldHasher<Sha256>>(transcript, verifier)
                    .unwrap()
            )
        }
    }
}
