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
        strides[i] = strides[i + 1]
            .checked_mul(shape[i + 1])
            .expect("Tensor: stride product overflowed usize");
    }
    strides
}

fn numel(shape: &[usize]) -> usize {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .expect("Tensor: numel overflowed usize")
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

    #[test]
    fn single_element_tensor() {
        let data = vec![42.0];
        let t = Tensor::new(&data, vec![1]);
        assert_eq!(t.rank(), 1);
        assert_eq!(t.shape(), &[1]);
        assert_eq!(t.strides(), &[1]);
        assert_eq!(t.offset(&[0]), 0);
        assert_eq!(t.data()[0], 42.0);
    }

    #[test]
    fn scalar_tensor_rank_zero() {
        let data = vec![7.0];
        let t = Tensor::new(&data, vec![]);
        assert_eq!(t.rank(), 0);
        assert!(t.shape().is_empty());
        assert!(t.strides().is_empty());
        assert_eq!(t.offset(&[]), 0);
    }

    #[test]
    fn rank_4_strides_are_row_major() {
        let data: Vec<f32> = (0..2 * 3 * 4 * 5).map(|x| x as f32).collect();
        let t = Tensor::new(&data, vec![2, 3, 4, 5]);
        assert_eq!(t.strides(), &[60, 20, 5, 1]);
        assert_eq!(t.offset(&[1, 2, 3, 4]), 60 + 40 + 15 + 4);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn offset_rejects_out_of_bounds_inner_dim() {
        let data = vec![0.0f32; 6];
        let t = Tensor::new(&data, vec![2, 3]);
        let _ = t.offset(&[0, 3]);
    }

    #[test]
    #[should_panic(expected = "requires rank-2")]
    fn row_panics_on_non_rank_2() {
        let data = vec![1.0f32, 2.0, 3.0];
        let t = Tensor::new(&data, vec![3]);
        let _ = t.row(0);
    }

    #[test]
    #[should_panic(expected = "Tensor::row")]
    fn row_panics_on_out_of_bounds() {
        let data = vec![0.0f32; 6];
        let t = Tensor::new(&data, vec![2, 3]);
        let _ = t.row(2);
    }

    #[test]
    #[should_panic(expected = "requires rank-2")]
    fn view_2d_panics_on_non_rank_2() {
        let data = vec![1.0f32, 2.0, 3.0];
        let t = Tensor::new(&data, vec![3]);
        let _ = t.view_2d(0..1);
    }

    #[test]
    #[should_panic(expected = "Tensor::view_2d")]
    fn view_2d_panics_on_out_of_bounds() {
        let data = vec![0.0f32; 6];
        let t = Tensor::new(&data, vec![2, 3]);
        let _ = t.view_2d(0..3);
    }

    #[test]
    fn view_2d_full_range_matches_original() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let t = Tensor::new(&data, vec![3, 4]);
        let v = t.view_2d(0..3);
        assert_eq!(v.shape(), &[3, 4]);
        assert_eq!(v.data(), t.data());
    }

    #[test]
    fn view_2d_empty_range() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let t = Tensor::new(&data, vec![3, 4]);
        let v = t.view_2d(1..1);
        assert_eq!(v.shape(), &[0, 4]);
        assert!(v.data().is_empty());
    }

    #[test]
    fn view_2d_single_row() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let t = Tensor::new(&data, vec![3, 4]);
        let v = t.view_2d(2..3);
        assert_eq!(v.shape(), &[1, 4]);
        assert_eq!(v.row(0), &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn owned_from_vec_preserves_data() {
        let t = OwnedTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "shape product")]
    fn owned_from_vec_panics_on_mismatched_len() {
        let _ = OwnedTensor::from_vec(vec![1.0, 2.0, 3.0], vec![2, 2]);
    }

    #[test]
    fn owned_zeros_has_correct_size() {
        let t = OwnedTensor::zeros(vec![4, 5]);
        assert_eq!(t.data().len(), 20);
        assert!(t.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn owned_as_view_round_trips_to_tensor() {
        let mut t = OwnedTensor::zeros(vec![3, 2]);
        for (i, v) in t.data_mut().iter_mut().enumerate() {
            *v = i as f32;
        }
        let v = t.as_view();
        assert_eq!(v.shape(), &[3, 2]);
        assert_eq!(v.offset(&[2, 1]), 5);
        assert_eq!(v.row(1), &[2.0, 3.0]);
    }

    #[test]
    fn empty_dim_tensor_has_zero_elements() {
        let data: Vec<f32> = Vec::new();
        let t = Tensor::new(&data, vec![0, 4]);
        assert_eq!(t.shape(), &[0, 4]);
        assert!(t.data().is_empty());
    }
}
