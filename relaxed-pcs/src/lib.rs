use std::{borrow::Borrow, collections::HashMap, marker::PhantomData};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{Config, LeafParam, TwoToOneParam},
    to_uncompressed_bytes, MerkleTree, Path,
};
use ark_ff::Field;
use ark_poly::{univariate, MultilinearExtension, Polynomial};
use ark_std::rand::Rng;

use gkr_protocol::{line, restrict_poly};

mod permutations;

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
    CompressionError,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Iterate over all possible values of a finite field.
pub trait IF: Field {
    type Values: IntoIterator<Item = Self>;

    fn all_values() -> Self::Values;

    fn all_multidimentional_values(m: usize) -> Vec<Vec<Self>> {
        let mut res: Vec<_> =
            permutations::permutations(&Self::all_values().into_iter().collect::<Vec<_>>(), m)
                .collect();
        res.sort();
        res
    }
}
pub struct Verifier<F: Field, P: Config<Leaf = [u8]>> {
    x: F,
    degree: usize,
    challenge_point: Vec<F>,
    line: Vec<univariate::SparsePolynomial<F>>,
    num_vars: usize,
    prover_univariate: Option<univariate::SparsePolynomial<F>>,
    merkle_root: P::InnerDigest,
    leaf_chr_params: LeafParam<P>,
    two_to_one_params: TwoToOneParam<P>,
    __f: PhantomData<F>,
}

pub enum VerifierMessage<F: Field> {
    RandomLineChallenge { b: Vec<F>, c: Vec<F> },
}

impl<F: Field, P: Config<Leaf = [u8]>> Verifier<F, P> {
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
            __f: PhantomData,
            merkle_root,
            leaf_chr_params,
            two_to_one_params,
        }
    }

    pub fn random_line<R: Rng>(&mut self, rng: &mut R) -> (Vec<F>, Vec<F>) {
        let b: Vec<F> = (0..self.num_vars).map(|_| F::rand(rng)).collect();
        let c: Vec<F> = (0..self.num_vars).map(|_| F::rand(rng)).collect();
        self.line = line(&b, &c);
        (b, c)
    }

    pub fn commited_univariate(&mut self, p: univariate::SparsePolynomial<F>) {
        assert_eq!(p.degree(), self.degree);
        self.prover_univariate = Some(p);
    }

    pub fn challenge_prover<R: Rng>(&mut self, rng: &mut R) -> Vec<F> {
        self.x = F::rand(rng);
        self.challenge_point = self
            .line
            .iter()
            .map(|poly| poly.evaluate(&self.x))
            .collect();
        self.challenge_point.clone()
    }

    pub fn verify_prover_reply(&self, path: Path<P>, leaf: F) -> Result<()> {
        let leaf_bytes = to_uncompressed_bytes!(leaf).map_err(|_| Error::CompressionError)?;

        path.verify(
            &self.leaf_chr_params,
            &self.two_to_one_params,
            &self.merkle_root,
            leaf_bytes,
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

pub struct Prover<F: Field, M: MultilinearExtension<F>, P: Config<Leaf = [u8]>> {
    tree: MerkleTree<P>,
    values_convenience_map: HashMap<Vec<F>, usize>,
    poly: M,
    values: Vec<F>,
    __p: PhantomData<P>,
    __f: PhantomData<F>,
}

impl<F: IF, M: MultilinearExtension<F>, P: Config<Leaf = [u8]>> Prover<F, M, P> {
    pub fn new(
        poly: M,

        leaf_chr_params: <<P as Config>::LeafHash as CRHScheme>::Parameters,
        two_to_one_params: <<P as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters,
    ) -> Result<Self> {
        let all_values = F::all_multidimentional_values(poly.num_vars());
        let all_poly_values: Result<Vec<_>> = all_values
            .iter()
            .map(|value| poly.evaluate(value).ok_or(Error::PolyEvalDimMismatch))
            .collect();

        let all_poly_values = all_poly_values?;

        let all_values_len = all_poly_values.len();
        let values: Result<Vec<_>> = all_poly_values
            .iter()
            .map(|value| to_uncompressed_bytes!(value).map_err(|_| Error::CompressionError))
            .chain(
                (all_values_len..all_values_len.next_power_of_two()).map(|_| {
                    to_uncompressed_bytes!(F::zero()).map_err(|_| Error::CompressionError)
                }),
            )
            .collect();

        let values = values?;

        let values_convenience_map = all_values
            .iter()
            .enumerate()
            .map(|(i, value)| (value.clone(), i))
            .collect();

        let tree: MerkleTree<P> =
            MerkleTree::new(&leaf_chr_params, &two_to_one_params, values.clone())?;

        let values: Result<Vec<_>> = values
            .into_iter()
            .map(|bytes| {
                F::deserialize_uncompressed(bytes.as_slice()).map_err(|_| Error::CompressionError)
            })
            .collect();

        let values = values?;
        Ok(Self {
            tree,
            poly,
            values_convenience_map,
            values,
            __p: PhantomData,
            __f: PhantomData,
        })
    }

    pub fn merkle_root(&self) -> P::InnerDigest {
        self.tree.root()
    }

    pub fn poly_restriction_to_line(&self, b: &[F], c: &[F]) -> univariate::SparsePolynomial<F> {
        restrict_poly(b, c, &self.poly)
    }

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
    use super::*;

    use ark_poly::DenseMultilinearExtension;
    use ark_std::test_rng;
    use pretty_assertions::assert_eq;

    use ark_crypto_primitives::{
        crh::{pedersen, TwoToOneCRHScheme},
        merkle_tree::{ByteDigestConverter, Config},
        CRHScheme, MerkleTree,
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

    struct JubJubMerkleTreeParams;

    impl Config for JubJubMerkleTreeParams {
        type Leaf = [u8];

        type LeafDigest = <LeafH as CRHScheme>::Output;
        type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
        type InnerDigest = <CompressH as TwoToOneCRHScheme>::Output;

        type LeafHash = LeafH;
        type TwoToOneHash = CompressH;
    }

    #[allow(unused)]
    type JubJubMerkleTree = MerkleTree<JubJubMerkleTreeParams>;

    #[derive(MontConfig)]
    #[modulus = "5"]
    #[generator = "2"]
    struct FrConfig;

    type Fp5 = Fp64<MontBackend<FrConfig, 1>>;

    impl IF for Fp5 {
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
        let prover: Prover<
            Fp5,
            DenseMultilinearExtension<ark_ff::Fp<MontBackend<FrConfig, 1>, 1>>,
            JubJubMerkleTreeParams,
        > = Prover::new(poly, leaf_chr_params.clone(), two_to_one_params.clone()).unwrap();

        let root = prover.merkle_root();

        let mut verifier: Verifier<Fp5, JubJubMerkleTreeParams> =
            Verifier::new(num_vars, degree, root, leaf_chr_params, two_to_one_params);

        let rand_line = verifier.random_line(rng);

        let restriction = prover.poly_restriction_to_line(&rand_line.0, &rand_line.1);

        let point = verifier.challenge_prover(rng);
        let (proof, value) = prover.challenge(point).unwrap();

        verifier.commited_univariate(restriction);

        verifier.verify_prover_reply(proof, value).unwrap();
    }
}
