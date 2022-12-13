use std::ops::Mul;

use ark_ff::{Field, Fp64, MontBackend, MontConfig, One};
use ark_std::{rand::Rng, test_rng, UniformRand};
use bitvec::slice::BitSlice;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use matrix_multiplication::G;
use sum_check_protocol::{Prover, SumCheckPolynomial};

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

const NUM_ITERS: u32 = 16;

pub fn benchmark_g_prover(c: &mut Criterion) {
    let rng = &mut test_rng();
    let mut group = c.benchmark_group("prover");
    group.sample_size(10);

    for p in 2..NUM_ITERS {
        let n = 2usize.pow(p);

        let a: Matrix<Fp5> = Matrix::new(n, rng);
        let b: Matrix<Fp5> = Matrix::new(n, rng);
        let mut point = u32_to_boolean_vec(2, p as usize);
        point.append(&mut u32_to_boolean_vec(2, p as usize));
        let g = G::new(
            p as usize,
            a.0.iter().flatten().cloned(),
            b.0.iter().flatten().cloned(),
            &point,
        );

        let num_vars = g.num_vars();

        group.throughput(Throughput::Elements(num_vars as u64));

        group.bench_with_input(BenchmarkId::new("prove", num_vars), &n, |b, _| {
            b.iter(|| {
                let mut r_j = Fp5::one();
                let mut prover = black_box(Prover::new(black_box(g.clone())));
                for j in 0..num_vars {
                    black_box(prover.round(black_box(r_j), black_box(j)));

                    r_j = Fp5::rand(rng);
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_g_prover);
criterion_main!(benches);
