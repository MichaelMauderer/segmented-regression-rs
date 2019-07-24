//! Segmented Linear Regression module
//!
//! Contains the implementation of segmented linear regression using OLS.
//!
//! # Usage
//!
//! ```
//! use segmented_regression_rs::SegmentedLinRegressor;
//! use rusty_machine::learning::SupModel;
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::linalg::Vector;
//! use assert_approx_eq::assert_approx_eq;
//!
//! let inputs = Vector::new(vec![-4.0, -3.0, -2.0, 2.0, 3.0, 4.0]);
//! let targets = Vector::new(vec![0.0, 0.0, 0.0, 4.0, 3.0, 2.0]);
//!
//! // We want two segments, one below zero and one above zero.
//! // Note that the boundaries will be included in the lower segment.
//! // That means this results in th segments [-inf, 0] and ]0, +inf].
//! let mut lin_mod = SegmentedLinRegressor::new(&[0.0]);
//!
//! // Train the model
//! lin_mod.train(&inputs, &targets).unwrap();
//!
//! // Now we'll predict a new point
//! let new_points = Vector::new(vec![-3.5,0.0, 3.5]);
//! let output = lin_mod.predict(&new_points).unwrap();
//!
//! // Hopefully we classified our new points correctly!
//! assert_approx_eq!(output[0], 0.0f64);
//! assert_approx_eq!(output[1], 0.0f64);
//! assert_approx_eq!(output[2], 2.5f64);
//! ```

use std::f64;

use rusty_machine::learning::error::Error;
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::Vector;
use rusty_machine::prelude::Matrix;

#[derive(Debug)]
struct Segment {
    bound_lower: f64,
    bound_upper: f64,
    model: Option<LinRegressor>,
}

impl Segment {
    /// Helper method that check whether the given value is within the boundaries of this segment.
    /// Segments are considered to exclude the lower bound and include the upper bound.
    fn in_segment(&self, value: f64) -> bool {
        self.bound_lower < value && value <= self.bound_upper
    }
}

/// Segmented Linear Regression Model.
#[derive(Debug)]
pub struct SegmentedLinRegressor {
    /// The segments for the segmented regression model.//
    /// Invariant: always have to be sorted from smallest to largest.
    segments: Vec<Segment>,
}

impl SegmentedLinRegressor {
    /// Create a new SegmentedLinRegressor with the given breakpoints.
    /// Cleans input: Ignores infinite, [subnormal][subnormal], or `NaN`.
    pub fn new(breakpoints: &[f64]) -> Self {

        // Ensure the breakpoints don't contain invalid values and are sorted.
        let mut breakpoints: Vec<f64> = {
            let mut filtered: Vec<f64> = breakpoints
                .iter()
                .filter_map(|f| { if f.is_normal() || (*f == 0.0f64) { Some(*f) } else { None } })
                .collect();
            // Sorting should never fail since we filtered all bad values.
            filtered.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // If it's not empty, the first value should be smaller (or equal) to the first one.
            debug_assert!(filtered.is_empty() || (filtered[0] <= filtered[filtered.len() - 1]));
            filtered
        };

        // Create a vec of f64s with all interval boundaries, including f64::MIN and f64::MAX for
        // the first and last interval.
        let interval_borders: Vec<f64> = {
            let mut interval_borders = Vec::with_capacity(breakpoints.len() + 2);
            interval_borders.push(f64::MIN);
            interval_borders.append(&mut breakpoints);
            interval_borders.push(f64::MAX);
            interval_borders
        };

        let segments = interval_borders.windows(2).map(|window| {
            debug_assert_eq!(window.len(), 2);
            Segment {
                bound_lower: window[0],
                bound_upper: window[1],
                model: None,
            }
        }).collect();

        SegmentedLinRegressor { segments }
    }
}

impl SupModel<Vector<f64>, Vector<f64>> for SegmentedLinRegressor {
    /// Predict output value from input data.
    ///
    /// Model must be trained before prediction can be made.
    fn predict(&self, inputs: &Vector<f64>) -> Result<Vector<f64>, Error> {
        // Iterate over each segment and choose the values that are in its range.
        let results: Vec<Result<Vector<(usize, f64)>, Error>> = self.segments.iter().map(|segment| {
            // Find the values for this segment
            let in_range_ix: Vec<usize> = inputs.iter().enumerate().filter_map(|(ix, value)| if segment.in_segment(*value) { Some(ix) } else { None }).collect();
            let segment_inputs_vec = inputs.select(&in_range_ix);
            let segment_inputs = Matrix::<f64>::new(segment_inputs_vec.size(), 1, segment_inputs_vec);

            // Compute the output and remember the input index.
            segment.model
                .as_ref()
                // We assume that all models have been initialised.
                .unwrap()
                .predict(&segment_inputs)
                .map(|segment_result| {
                    let tuples: Vec<(usize, f64)> = in_range_ix.into_iter().zip(segment_result).collect();
                    Vector::from(tuples)
                })
        }).collect();

        // Collect the values from each segment and put them at the right index.
        // If we encounter any error we return the first error encountered.
        let mut result: Vector<f64> = Vector::zeros(inputs.size());
        for segment_result in results {
            segment_result?.iter().for_each(|(ix, value)| {
                result[*ix] = *value
            })
        }

        Ok(result)
    }

    /// Train the linear regression model.
   ///
   /// Takes training data and output values as input.
   ///
   /// # Examples
   ///
   /// ```
   /// use segmented_regression_rs::SegmentedLinRegressor;
   /// use rusty_machine::linalg::Matrix;
   /// use rusty_machine::linalg::Vector;
   /// use rusty_machine::learning::SupModel;
   ///
   /// let mut lin_mod = SegmentedLinRegressor::new(&[0.0]);
   /// let inputs = Vector::new(vec![-4.0, -3.0, -2.0, 2.0, 3.0, 4.0]);
   /// let targets = Vector::new(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0]);
   ///
   /// lin_mod.train(&inputs, &targets).unwrap();
   /// ```
    fn train(&mut self, inputs: &Vector<f64>, targets: &Vector<f64>) -> Result<(), Error> {
        let results: Vec<Result<(), Error>> = self.segments.iter_mut().map(|segment| {
            // Find the values for this segment
            let in_range_ix: Vec<usize> = inputs.iter().enumerate().filter_map(|(ix, value)| if segment.in_segment(*value) { Some(ix) } else { None }).collect();
            let segment_inputs_vec = inputs.select(&in_range_ix);
            let segment_inputs = Matrix::<f64>::new(segment_inputs_vec.size(), 1, segment_inputs_vec);
            let segment_targets = targets.select(&in_range_ix);

            // Train a LinReg for this segment;
            let mut lin_reg = LinRegressor::default();
            lin_reg.train(&segment_inputs, &segment_targets)?;
            segment.model = Some(lin_reg);
            Ok(())
        }).collect();

        // Assert that all segments have been initialised.
        debug_assert_eq!(self.segments.iter().filter(|s| s.model.is_none()).count(), 0);

        // Check if there are errors and if so, return the first error found.
        for result in results {
            result?
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use rusty_machine::learning::SupModel;
    use rusty_machine::linalg::Vector;

    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_unsorted() {

        // Sorted should be [-1.0, 0.0, 2.0]
        let mut lin_mod = SegmentedLinRegressor::new(&[2.0, 0.0, -1.0]);
        assert_eq!(lin_mod.segments.len(), 4);

        assert_approx_eq!(lin_mod.segments[0].bound_upper, -1.0);
        assert_approx_eq!(lin_mod.segments[1].bound_lower, -1.0);
        assert_approx_eq!(lin_mod.segments[1].bound_upper, 0.0);
        assert_approx_eq!(lin_mod.segments[2].bound_lower, 0.0);
        assert_approx_eq!(lin_mod.segments[2].bound_upper, 2.0);
        assert_approx_eq!(lin_mod.segments[3].bound_lower, 2.0);


        let inputs = Vector::new(vec![-3.0, -2.0, -0.25, -0.5, 1.0, 1.7, 3.0, 4.0]);
        let targets = Vector::new(vec![-10., -10.0, 3.0, 3.0, 99.0, 99.0, 3.0, 4.0]);

        lin_mod.train(&inputs, &targets).unwrap();

        let new_points = Vector::new(vec![-10.0, -0.5, 0.5, 50.0]);
        let output = lin_mod.predict(&new_points).unwrap();

        assert_approx_eq!(output[0], -10.0f64);
        assert_approx_eq!(output[1], 3.0f64);
        assert_approx_eq!(output[2], 99.0f64);
        assert_approx_eq!(output[3], 50.0f64);
    }

    #[test]
    fn test_float_filter() {
        let lin_mod = SegmentedLinRegressor::new(&[f64::INFINITY, 0.0, f64::NEG_INFINITY, f64::NAN]);
        assert_eq!(lin_mod.segments.len(), 2);

        assert_approx_eq!(lin_mod.segments[0].bound_upper, 0.0);
        assert_approx_eq!(lin_mod.segments[1].bound_lower, 0.0);
    }
}