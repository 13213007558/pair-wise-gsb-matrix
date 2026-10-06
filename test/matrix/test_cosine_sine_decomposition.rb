# frozen_string_literal: false
require 'test/unit'
require 'matrix'

class TestMatrixCosineSineDecomposition < Test::Unit::TestCase
  def rotation(theta)
    Matrix[[Math.cos(theta), -Math.sin(theta)],
           [Math.sin(theta),  Math.cos(theta)]]
  end

  def assert_rebuilds(matrix, row_count, column_count)
    decomposition = matrix.csd(row_count, column_count)
    rebuilt = decomposition.u * decomposition.theta * decomposition.v.transpose

    assert_equal matrix.row_count, decomposition.u.row_count
    assert_equal matrix.column_count, decomposition.v.column_count
    assert_equal matrix.row_count, decomposition.theta.row_count
    assert_equal matrix.column_count, decomposition.theta.column_count
    assert_in_delta_array(matrix, rebuilt, 1e-10)

    identity = Matrix.identity(matrix.row_count)
    assert_in_delta_array(identity, decomposition.u.transpose * decomposition.u, 1e-10)
    assert_in_delta_array(identity, decomposition.v.transpose * decomposition.v, 1e-10)
  end

  def assert_in_delta_array(expected, actual, delta)
    expected.row_count.times do |row|
      expected.column_count.times do |column|
        assert_in_delta expected[row, column], actual[row, column], delta
      end
    end
  end

  def test_returns_shared_bases_and_block_sizes_for_unbalanced_partition
    matrix = Matrix.identity(5)
    decomposition = matrix.csd(2, 3)

    assert_equal 5, decomposition.u.row_count
    assert_equal 5, decomposition.v.row_count
    expected_top_left = Matrix.build(2, 3) { |row, column| row == column ? 1 : 0 }
    assert_equal expected_top_left, decomposition.top_left
    assert_equal Matrix.zero(2, 2), decomposition.top_right
    expected_bottom_right = Matrix.build(3, 2) { |row, column| row == column ? 1 : 0 }
    assert_equal expected_bottom_right, decomposition.bottom_right
    assert_equal [2, 3], [decomposition.top_left.row_count, decomposition.top_left.column_count]
    assert_equal [3, 3], [decomposition.bottom_left.row_count, decomposition.bottom_left.column_count]
    assert_rebuilds matrix, 2, 3
  end

  def test_single_rotation_uses_one_shared_angle
    angle = 0.7
    matrix = rotation(angle)
    decomposition = matrix.csd(1, 1)

    assert_equal [angle], decomposition.rotation_angles
    assert_equal [angle], decomposition.principal_angles
    assert_equal Matrix.build(1) { Math.cos(angle) }, decomposition.top_left
    assert_equal(-Matrix.build(1) { Math.sin(angle) }, decomposition.top_right)
    assert_equal Matrix.build(1) { Math.sin(angle) }, decomposition.bottom_left
    assert_equal Matrix.build(1) { Math.cos(angle) }, decomposition.bottom_right
    assert_rebuilds matrix, 1, 1
  end

  def test_exact_zero_and_right_angle_complete_identity_and_zero_blocks
    identity = Matrix.identity(4)
    right_angle = Matrix[[0, -1], [1, 0]]

    assert_rebuilds identity, 2, 2
    assert_rebuilds right_angle, 1, 1
    assert_rebuilds identity, 0, 4
    assert_rebuilds identity, 4, 0
    assert_rebuilds identity, 0, 0
  end

  def test_all_partition_positions_for_identity_include_rectangular_zero_blocks
    matrix = Matrix.identity(4)
    (0..4).each do |row_count|
      (0..4).each do |column_count|
        decomposition = matrix.csd(row_count, column_count)
        assert_equal [row_count, column_count],
                     [decomposition.top_left.row_count, decomposition.top_left.column_count]
        assert_equal [4 - row_count, column_count],
                     [decomposition.bottom_left.row_count, decomposition.bottom_left.column_count]
        assert_equal [row_count, 4 - column_count],
                     [decomposition.top_right.row_count, decomposition.top_right.column_count]
        assert_equal [4 - row_count, 4 - column_count],
                     [decomposition.bottom_right.row_count, decomposition.bottom_right.column_count]
        assert_rebuilds matrix, row_count, column_count
      end
    end
  end

  def test_rejects_non_orthogonal_matrices
    matrix = Matrix[[1.0, 0.2], [0.0, 1.0]]
    assert_raise(Matrix::ErrOperationNotDefined) do
      matrix.csd(1, 1)
    end
  end
end
