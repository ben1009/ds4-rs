//! Minimal row-major f32 tensor views.
//!
//! See rfcs/0002-forward-pass.md §2. No autograd, no broadcasting — just a
//! shape-carrying `&[f32]` view plus a few safe indexing helpers so ops
//! don't hand-roll stride math.

use std::ops::Range;

/// Borrowed f32 view with shape + row-major strides.
#[derive(Clone, Debug)]
pub struct Tensor<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
    strides: Vec<usize>,
}

/// Owned f32 buffer with shape. Used for scratch activations.
#[derive(Clone, Debug)]
pub struct OwnedTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl<'a> Tensor<'a> {
    /// Row-major view. Panics if `data.len() != shape.iter().product()`.
    pub fn new(data: &'a [f32], shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            numel(&shape),
            "Tensor::new: data.len() = {} but shape product = {}",
            data.len(),
            numel(&shape),
        );
        let strides = row_major_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn data(&self) -> &[f32] {
        self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Flat offset for a coordinate. Panics on rank or bounds mismatch.
    pub fn offset(&self, coords: &[usize]) -> usize {
        assert_eq!(
            coords.len(),
            self.shape.len(),
            "Tensor::offset: rank mismatch ({} vs {})",
            coords.len(),
            self.shape.len(),
        );
        let mut off = 0;
        for (i, &c) in coords.iter().enumerate() {
            assert!(
                c < self.shape[i],
                "Tensor::offset: dim {i} index {c} out of bounds for size {}",
                self.shape[i],
            );
            off += c * self.strides[i];
        }
        off
    }

    /// Slice one row from a 2-D tensor.
    pub fn row(&self, i: usize) -> &'a [f32] {
        assert_eq!(self.shape.len(), 2, "Tensor::row: requires rank-2 tensor");
        assert!(
            i < self.shape[0],
            "Tensor::row: index {i} >= {}",
            self.shape[0]
        );
        let stride = self.strides[0];
        &self.data[i * stride..i * stride + self.shape[1]]
    }

    /// Slice a row range from a 2-D tensor, returning a new view.
    pub fn view_2d(&self, rows: Range<usize>) -> Tensor<'a> {
        assert_eq!(
            self.shape.len(),
            2,
            "Tensor::view_2d: requires rank-2 tensor",
        );
        assert!(
            rows.end <= self.shape[0],
            "Tensor::view_2d: end {} > rows {}",
            rows.end,
            self.shape[0],
        );
        let stride = self.strides[0];
        let start = rows.start * stride;
        let end = rows.end * stride;
        Tensor::new(
            &self.data[start..end],
            vec![rows.end - rows.start, self.shape[1]],
        )
    }
}

impl OwnedTensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = numel(&shape);
        let strides = row_major_strides(&shape);
        Self {
            data: vec![0.0; n],
            shape,
            strides,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            numel(&shape),
            "OwnedTensor::from_vec: data.len() = {} but shape product = {}",
            data.len(),
            numel(&shape),
        );
        let strides = row_major_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn as_view(&self) -> Tensor<'_> {
        Tensor {
            data: &self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_derives_row_major_strides() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let t = Tensor::new(&data, vec![3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.strides(), &[4, 1]);
        assert_eq!(t.rank(), 2);
    }

    #[test]
    fn offset_matches_row_major_layout() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = Tensor::new(&data, vec![2, 3, 4]);
        // (1, 2, 3) => 1*12 + 2*4 + 3 = 23
        assert_eq!(t.offset(&[1, 2, 3]), 23);
        assert_eq!(t.data()[t.offset(&[1, 2, 3])], 23.0);
        // (0, 0, 0) => 0
        assert_eq!(t.offset(&[0, 0, 0]), 0);
    }

    #[test]
    #[should_panic(expected = "rank mismatch")]
    fn offset_rejects_rank_mismatch() {
        let data = vec![0.0f32; 6];
        let t = Tensor::new(&data, vec![2, 3]);
        let _ = t.offset(&[1]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn offset_rejects_out_of_bounds() {
        let data = vec![0.0f32; 6];
        let t = Tensor::new(&data, vec![2, 3]);
        let _ = t.offset(&[2, 0]);
    }

    #[test]
    fn row_returns_contiguous_slice() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let t = Tensor::new(&data, vec![3, 4]);
        assert_eq!(t.row(0), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(t.row(2), &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn view_2d_slices_rows() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let t = Tensor::new(&data, vec![3, 4]);
        let mid = t.view_2d(1..3);
        assert_eq!(mid.shape(), &[2, 4]);
        assert_eq!(mid.row(0), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(mid.row(1), &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    #[should_panic(expected = "shape product")]
    fn new_panics_on_mismatched_len() {
        let data = vec![0.0f32; 5];
        let _ = Tensor::new(&data, vec![2, 3]);
    }

    #[test]
    fn owned_tensor_zeros_and_view() {
        let mut t = OwnedTensor::zeros(vec![2, 3]);
        t.data_mut()[4] = 7.0;
        assert_eq!(t.data(), &[0.0, 0.0, 0.0, 0.0, 7.0, 0.0]);
        let v = t.as_view();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.row(1), &[0.0, 7.0, 0.0]);
    }

    #[test]
    fn rank_1_tensor() {
        let data = vec![1.0, 2.0, 3.0];
        let t = Tensor::new(&data, vec![3]);
        assert_eq!(t.rank(), 1);
        assert_eq!(t.strides(), &[1]);
        assert_eq!(t.offset(&[2]), 2);
    }
}
