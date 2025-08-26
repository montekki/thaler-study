#![deny(unused_crate_dependencies)]
#![deny(missing_docs)]

//! The implementation of the Relaxed PCS protocol.

use std::collections::HashMap;

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{Config, LeafParam, MerkleTree, Path, TwoToOneParam},
};
use ark_ff::Field;
use ark_poly::{univariate, MultilinearExtension, Polynomial};
use ark_std::rand::Rng;

use gkr_protocol::{line, restrict_poly};

mod permutations;

/// Crate error type.
#[allow(missing_docs)]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    ArkCryptoPrimitivesError(#[from] ark_crypto_primitives::Error),

    #[error("Evaluation does not match leaf {0} {1}")]
    EvalMismatch(String, String),

    #[error("Prover not commited a polynomial")]
    NoProverPoly,

    #[error("Poly evaluation dimention mismatch")]
    PolyEvalDimMismatch,

    #[error("Compression error")]
    ToBytesError,

    #[error("Prover claim degree mismatch")]
    DegreeMismatch,
}

/// Crate `Result` type.
pub type Result<T> = std::result::Result<T, Error>;

/// Iterate over all possible values of a finite field.
pub trait IF: Field {
    /// Type of the values.
    type Values: IntoIterator<Item = Self>;

    /// Get all values of the type.
    fn all_values() -> Self::Values;

    /// Get all permutations of the values of the type.
    fn all_multidimentional_values(m: usize) -> Vec<Vec<Self>> {
        let mut res: Vec<_> =
            permutations::permutations(&Self::all_values().into_iter().collect::<Vec<_>>(), m)
                .collect();
        res.sort();
        res
    }
}

/// The Verifier in the Relaxed PCS protocol.
pub struct Verifier<F: Field, P: Config<Leaf = F>> {
    x: F,
    degree: usize,
    challenge_point: Vec<F>,
    line: Vec<univariate::SparsePolynomial<F>>,
    num_vars: usize,
    prover_univariate: Option<univariate::SparsePolynomial<F>>,
    merkle_root: P::InnerDigest,
    leaf_chr_params: LeafParam<P>,
    two_to_one_params: TwoToOneParam<P>,
}

impl<F: Field, P: Config<Leaf = F>> Verifier<F, P> {
    /// Create a new Verifier.
    pub fn new(
        num_vars: usize,
        degree: usize,
        merkle_root: P::InnerDigest,
        leaf_chr_params: <<P as Config>::LeafHash as CRHScheme>::Parameters,
        two_to_one_params: <<P as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters,
    ) -> Self {
        Self {
            x: F::zero(),
            degree: degree * num_vars,
            challenge_point: vec![],
            line: vec![],
            prover_univariate: None,
            num_vars,
            merkle_root,
            leaf_chr_params,
            two_to_one_params,
        }
    }

    /// Generate a random line to challenge the `Prover`.
    pub fn random_line<R: Rng>(&mut self, rng: &mut R) -> (Vec<F>, Vec<F>) {
        let b: Vec<F> = (0..self.num_vars).map(|_| F::rand(rng)).collect();
        let c: Vec<F> = (0..self.num_vars).map(|_| F::rand(rng)).collect();
        self.line = line(&b, &c);
        (b, c)
    }

    /// Receive the commited univariate polynomial from Prover.
    pub fn commited_univariate(&mut self, p: univariate::SparsePolynomial<F>) -> Result<()> {
        if p.degree() != self.degree {
            return Err(Error::DegreeMismatch);
        }
        self.prover_univariate = Some(p);
        Ok(())
    }

    /// Challenge the prover at some point.
    pub fn challenge_prover<R: Rng>(&mut self, rng: &mut R) -> Vec<F> {
        self.x = F::rand(rng);
        self.challenge_point = self
            .line
            .iter()
            .map(|poly| poly.evaluate(&self.x))
            .collect();
        self.challenge_point.clone()
    }

    /// Verify the prover's reply.
    pub fn verify_prover_reply(&self, path: Path<P>, leaf: F) -> Result<()> {
        path.verify(
            &self.leaf_chr_params,
            &self.two_to_one_params,
            &self.merkle_root,
            leaf,
        )?;

        let eval = self
            .prover_univariate
            .as_ref()
            .ok_or(Error::NoProverPoly)?
            .evaluate(&self.x);
        if leaf != eval {
            return Err(Error::EvalMismatch(
                format!("{:?}", leaf),
                format!("{:?}", eval),
            ));
        }
        Ok(())
    }
}

/// Prover in the Relaxed PCS protocol.
pub struct Prover<F: Field, M: MultilinearExtension<F>, P: Config<Leaf = F>> {
    tree: MerkleTree<P>,
    values_convenience_map: HashMap<Vec<F>, usize>,
    poly: M,
    values: Vec<F>,
}

impl<F: IF + AsRef<P::Leaf>, M: MultilinearExtension<F>, P: Config<Leaf = F>> Prover<F, M, P> {
    /// Create a new Prover.
    pub fn new(
        poly: M,
        leaf_chr_params: <<P as Config>::LeafHash as CRHScheme>::Parameters,
        two_to_one_params: <<P as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters,
    ) -> Result<Self> {
        let all_values = F::all_multidimentional_values(poly.num_vars());
        let all_poly_values: Vec<_> = all_values
            .iter()
            .map(|value| poly.evaluate(value))
            .collect();

        let all_values_len = all_poly_values.len();
        let values: Vec<_> = all_poly_values
            .iter()
            .cloned()
            .chain((all_values_len..all_values_len.next_power_of_two()).map(|_| F::zero()))
            .collect();

        let values_convenience_map = all_values
            .iter()
            .enumerate()
            .map(|(i, value)| (value.clone(), i))
            .collect();

        let tree: MerkleTree<P> =
            MerkleTree::new(&leaf_chr_params, &two_to_one_params, values.clone())?;

        Ok(Self {
            tree,
            poly,
            values_convenience_map,
            values,
        })
    }

    /// Get the merkle root.
    pub fn merkle_root(&self) -> P::InnerDigest {
        self.tree.root()
    }

    /// Restrict to line.
    pub fn poly_restriction_to_line(&self, b: &[F], c: &[F]) -> univariate::SparsePolynomial<F> {
        restrict_poly(b, c, &self.poly)
    }

    /// Challenge
    pub fn challenge(&self, point: Vec<F>) -> Result<(Path<P>, F)> {
        let point_index = self.values_convenience_map.get(&point).unwrap();
        Ok((
            self.tree.generate_proof(*point_index)?,
            self.values[*point_index],
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Borrow, marker::PhantomData};

    use super::*;

    use ark_poly::DenseMultilinearExtension;
    use ark_std::test_rng;

    use ark_crypto_primitives::{
        crh::{pedersen, CRHScheme, TwoToOneCRHScheme},
        merkle_tree::{ByteDigestConverter, Config, MerkleTree},
        to_uncompressed_bytes,
    };
    use ark_ed_on_bls12_381::EdwardsProjective as JubJub;
    use ark_ff::{Fp64, MontBackend, MontConfig};

    #[derive(Clone)]
    pub(super) struct Window4x256;
    impl pedersen::Window for Window4x256 {
        const WINDOW_SIZE: usize = 4;
        const NUM_WINDOWS: usize = 256;
    }

    type LeafH = pedersen::CRH<JubJub, Window4x256>;
    type CompressH = pedersen::TwoToOneCRH<JubJub, Window4x256>;

    struct CHROverField<F> {
        __f: PhantomData<F>,
    }

    impl<F: Field> CRHScheme for CHROverField<F> {
        type Input = F;

        type Output = <LeafH as CRHScheme>::Output;

        type Parameters = <LeafH as CRHScheme>::Parameters;

        fn setup<R: Rng>(
            r: &mut R,
        ) -> std::result::Result<Self::Parameters, ark_crypto_primitives::Error> {
            LeafH::setup(r)
        }

        fn evaluate<T: Borrow<Self::Input>>(
            parameters: &Self::Parameters,
            input: T,
        ) -> std::result::Result<Self::Output, ark_crypto_primitives::Error> {
            let bytes = to_uncompressed_bytes!(input)?;
            LeafH::evaluate(parameters, bytes.as_ref())
        }
    }

    struct JubJubMerkleTreeParamsLocalFp5;

    impl Config for JubJubMerkleTreeParamsLocalFp5 {
        type Leaf = LocalFp5;

        type LeafDigest = <LeafH as CRHScheme>::Output;
        type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
        type InnerDigest = <CompressH as TwoToOneCRHScheme>::Output;

        type LeafHash = CHROverField<LocalFp5>;
        type TwoToOneHash = CompressH;
    }

    #[allow(unused)]
    type JubJubMerkleTree = MerkleTree<JubJubMerkleTreeParamsLocalFp5>;

    #[derive(MontConfig)]
    #[modulus = "5"]
    #[generator = "2"]
    struct FrConfig;

    // Simple newtype to satisfy AsRef requirement for arkworks 0.5
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Fp5(Fp64<MontBackend<FrConfig, 1>>);

    impl From<u32> for Fp5 {
        fn from(val: u32) -> Self {
            Fp5(Fp64::from(val))
        }
    }

    impl AsRef<Fp5> for Fp5 {
        fn as_ref(&self) -> &Fp5 {
            self
        }
    }

    // Forward all Field operations to inner type
    impl Field for Fp5 {
        type BasePrimeField = Fp64<MontBackend<FrConfig, 1>>;

        fn extension_degree() -> u64 { 1 }
        
        fn to_base_prime_field_elements(&self) -> impl Iterator<Item = Self::BasePrimeField> {
            std::iter::once(self.0)
        }

        fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
            elems.get(0).map(|&x| Fp5(x))
        }

        fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
            Fp5(elem)
        }

        fn characteristic() -> &'static [u64] {
            self.0.characteristic()
        }

        fn from_random_bytes_with_flags<F: ark_ff::Flags>(bytes: &[u8]) -> Option<(Self, F)> {
            self.0.from_random_bytes_with_flags(bytes).map(|(f, flag)| (Fp5(f), flag))
        }

        fn square(&self) -> Self { Fp5(self.0.square()) }
        fn square_in_place(&mut self) -> &mut Self { self.0.square_in_place(); self }
        fn inverse(&self) -> Option<Self> { self.0.inverse().map(Fp5) }
        fn inverse_in_place(&mut self) -> Option<&mut Self> { self.0.inverse_in_place().map(|_| self) }
        fn frobenius_map_in_place(&mut self, power: usize) { self.0.frobenius_map_in_place(power); }
    }

    // All arithmetic operations
    impl std::ops::Add for Fp5 {
        type Output = Self;
        fn add(self, other: Self) -> Self { Fp5(self.0 + other.0) }
    }
    impl std::ops::AddAssign for Fp5 {
        fn add_assign(&mut self, other: Self) { self.0 += other.0; }
    }
    impl std::ops::Sub for Fp5 {
        type Output = Self;
        fn sub(self, other: Self) -> Self { Fp5(self.0 - other.0) }
    }
    impl std::ops::SubAssign for Fp5 {
        fn sub_assign(&mut self, other: Self) { self.0 -= other.0; }
    }
    impl std::ops::Mul for Fp5 {
        type Output = Self;
        fn mul(self, other: Self) -> Self { Fp5(self.0 * other.0) }
    }
    impl std::ops::MulAssign for Fp5 {
        fn mul_assign(&mut self, other: Self) { self.0 *= other.0; }
    }
    impl std::ops::Div for Fp5 {
        type Output = Self;
        fn div(self, other: Self) -> Self { Fp5(self.0 / other.0) }
    }
    impl std::ops::DivAssign for Fp5 {
        fn div_assign(&mut self, other: Self) { self.0 /= other.0; }
    }
    impl std::ops::Neg for Fp5 {
        type Output = Self;
        fn neg(self) -> Self { Fp5(-self.0) }
    }

    impl ark_std::rand::distributions::Distribution<Fp5> for ark_std::rand::distributions::Standard {
        fn sample<R: ark_std::rand::Rng + ?Sized>(&self, rng: &mut R) -> Fp5 {
            Fp5(Fp64::rand(rng))
        }
    }

    // Local re-definition of Fp5 to work around orphan rules
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct LocalFp5(Fp64<MontBackend<FrConfig, 1>>);

    impl From<u32> for LocalFp5 {
        fn from(val: u32) -> Self {
            LocalFp5(Fp64::from(val))
        }
    }

    impl AsRef<LocalFp5> for LocalFp5 {
        fn as_ref(&self) -> &LocalFp5 {
            self
        }
    }

    // Delegate Field implementation to inner type
    impl Field for LocalFp5 {
        type BasePrimeField = Fp64<MontBackend<FrConfig, 1>>;

        fn extension_degree() -> u64 { 1 }
        
        fn to_base_prime_field_elements(&self) -> impl Iterator<Item = Self::BasePrimeField> {
            std::iter::once(self.0)
        }

        fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
            elems.get(0).map(|&x| LocalFp5(x))
        }

        fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
            LocalFp5(elem)
        }

        fn characteristic() -> &'static [u64] {
            Fp64::<MontBackend<FrConfig, 1>>::characteristic()
        }

        fn from_random_bytes_with_flags<F: ark_ff::Flags>(bytes: &[u8]) -> Option<(Self, F)> {
            Fp64::<MontBackend<FrConfig, 1>>::from_random_bytes_with_flags(bytes).map(|(f, flag)| (LocalFp5(f), flag))
        }

        fn square(&self) -> Self { LocalFp5(self.0.square()) }
        fn square_in_place(&mut self) -> &mut Self { self.0.square_in_place(); self }
        fn inverse(&self) -> Option<Self> { self.0.inverse().map(LocalFp5) }
        fn inverse_in_place(&mut self) -> Option<&mut Self> { self.0.inverse_in_place().map(|_| self) }
        fn frobenius_map_in_place(&mut self, power: usize) { self.0.frobenius_map_in_place(power); }
    }

    // Arithmetic operations
    impl std::ops::Add for LocalFp5 {
        type Output = Self;
        fn add(self, other: Self) -> Self { LocalFp5(self.0 + other.0) }
    }
    impl std::ops::AddAssign for LocalFp5 {
        fn add_assign(&mut self, other: Self) { self.0 += other.0; }
    }
    impl std::ops::Sub for LocalFp5 {
        type Output = Self;
        fn sub(self, other: Self) -> Self { LocalFp5(self.0 - other.0) }
    }
    impl std::ops::SubAssign for LocalFp5 {
        fn sub_assign(&mut self, other: Self) { self.0 -= other.0; }
    }
    impl std::ops::Mul for LocalFp5 {
        type Output = Self;
        fn mul(self, other: Self) -> Self { LocalFp5(self.0 * other.0) }
    }
    impl std::ops::MulAssign for LocalFp5 {
        fn mul_assign(&mut self, other: Self) { self.0 *= other.0; }
    }
    impl std::ops::Div for LocalFp5 {
        type Output = Self;
        fn div(self, other: Self) -> Self { LocalFp5(self.0 / other.0) }
    }
    impl std::ops::DivAssign for LocalFp5 {
        fn div_assign(&mut self, other: Self) { self.0 /= other.0; }
    }
    impl std::ops::Neg for LocalFp5 {
        type Output = Self;
        fn neg(self) -> Self { LocalFp5(-self.0) }
    }

    impl ark_std::rand::distributions::Distribution<LocalFp5> for ark_std::rand::distributions::Standard {
        fn sample<R: ark_std::rand::Rng + ?Sized>(&self, rng: &mut R) -> LocalFp5 {
            LocalFp5(Fp64::rand(rng))
        }
    }

    impl IF for LocalFp5 {
        type Values = Vec<Self>;

        fn all_values() -> Self::Values {
            (0..5u32).map(From::from).collect()
        }
    }

    #[test]
    fn it_works() {
        let v = Fp5::all_values();
        let rng = &mut test_rng();
        let num_vars = 2;
        let degree = 1;
        let poly = DenseMultilinearExtension::rand(num_vars, rng);

        assert_eq!(v, (0..5u32).map(From::from).collect::<Vec<_>>());

        let leaf_chr_params = <LeafH as CRHScheme>::setup(rng).unwrap();
        let two_to_one_params = <CompressH as TwoToOneCRHScheme>::setup(rng).unwrap();
        let prover: Prover<Fp5, DenseMultilinearExtension<Fp5>, JubJubMerkleTreeParamsFp5> =
            Prover::new(poly, leaf_chr_params.clone(), two_to_one_params.clone()).unwrap();

        let root = prover.merkle_root();

        let mut verifier: Verifier<Fp5, JubJubMerkleTreeParamsFp5> =
            Verifier::new(num_vars, degree, root, leaf_chr_params, two_to_one_params);

        let rand_line = verifier.random_line(rng);

        let restriction = prover.poly_restriction_to_line(&rand_line.0, &rand_line.1);

        let point = verifier.challenge_prover(rng);
        let (proof, value) = prover.challenge(point).unwrap();

        verifier.commited_univariate(restriction).unwrap();

        verifier.verify_prover_reply(proof, value).unwrap();
    }
}
