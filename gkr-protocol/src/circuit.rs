use std::ops::{Add, Mul};

use ark_ff::Field;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};

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
    pub fn new(layers: Vec<CircuitLayer>, num_inputs: usize) -> Self {
        Self { layers, num_inputs }
    }

    pub fn num_vars_at(&self, layer: usize) -> Option<usize> {
        let num_gates = if let Some(layer) = self.layers.get(layer) {
            layer.len()
        } else if layer == self.layers.len() {
            self.num_inputs
        } else {
            return None;
        };

        Some((num_gates as u64).trailing_zeros() as usize)
    }

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

    pub fn layers(&self) -> &[CircuitLayer] {
        &self.layers
    }

    pub fn num_outputs(&self) -> usize {
        self.layers[0].layer.len()
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    pub fn add_i_ext<F: Field>(&self, r_i: &[F], i: usize) -> DenseMultilinearExtension<F> {
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

    pub fn mul_i_ext<F: Field>(&self, r_i: &[F], i: usize) -> DenseMultilinearExtension<F> {
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
#[cfg(test)]
pub(crate) fn circuit_from_book() -> Circuit {
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
#[cfg(test)]
mod tests {
    use super::circuit_from_book;

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
}
