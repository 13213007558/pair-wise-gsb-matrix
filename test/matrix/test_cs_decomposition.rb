# frozen_string_literal: false
require 'test/unit'
require 'matrix'

class TestMatrixCSDecomposition < Test::Unit::TestCase
  EPS = 1e-9

  def random_orthogonal(n, seed)
    rng = Random.new(seed)
    a = Matrix.build(n) { rng.rand - 0.5 }
    (a + a.t).eigensystem.eigenvector_matrix
  end

  def block_diag(a, b)
    Matrix.hstack(Matrix.vstack(a, Matrix.zero(b.row_count, a.column_count)),
                  Matrix.vstack(Matrix.zero(a.row_count, b.column_count), b))
  end

  def assert_matrix_close(expected, actual, tol = EPS)
    assert_equal(expected.row_count, actual.row_count)
    assert_equal(expected.column_count, actual.column_count)
    assert(actual.each_with_index.all? { |x, i, j| (x - expected[i, j]).abs < tol },
           "matrices differ:\nexpected: #{expected}\nactual: #{actual}")
  end

  def test_reconstruction_and_factors
    [[4, 2, 2], [7, 2, 4], [7, 5, 3], [8, 3, 6], [6, 1, 5], [9, 4, 4],
     [10, 8, 2], [5, 4, 1], [2, 1, 1]].each_with_index do |(n, p, q), idx|
      m = random_orthogonal(n, 42 + idx)
      cs = m.cs_decomposition(p, q)
      u, d, v = cs.to_ary

      assert_matrix_close(m, u * d * v.t)
      assert_matrix_close(Matrix.identity(n), u * u.t)
      assert_matrix_close(Matrix.identity(n), v * v.t)

      # U and V are block diagonal, sharing one basis for all four blocks
      assert_equal(Matrix.zero(p, n - p), u.minor(0...p, p...n))
      assert_equal(Matrix.zero(n - p, p), u.minor(p...n, 0...p))
      assert_equal(Matrix.zero(q, n - q), v.minor(0...q, q...n))
      assert_equal(Matrix.zero(n - q, q), v.minor(q...n, 0...q))

      assert_equal([p, q].min, cs.theta.size)
      assert(cs.theta.all? { |t| t >= 0 && t <= Math::PI / 2 })
      assert(cs.theta.each_cons(2).all? { |a, b| a <= b })
      cs.theta.each_with_index do |t, i|
        assert_in_delta(Math.cos(t), cs.cosines[i], EPS)
        assert_in_delta(Math.sin(t), cs.sines[i], EPS)
      end
    end
  end

  def test_factor_dimensions
    cs = random_orthogonal(7, 7).cs_decomposition(2, 4)
    assert_equal([2, 2], [cs.u1.row_count, cs.u1.column_count])
    assert_equal([5, 5], [cs.u2.row_count, cs.u2.column_count])
    assert_equal([4, 4], [cs.v1.row_count, cs.v1.column_count])
    assert_equal([3, 3], [cs.v2.row_count, cs.v2.column_count])
    assert_equal([7, 7], [cs.u.row_count, cs.u.column_count])
    assert_equal([7, 7], [cs.v.row_count, cs.v.column_count])
    assert_equal([7, 7], [cs.cs.row_count, cs.cs.column_count])
    assert_equal(2, cs.theta.size)
    assert_equal([cs.u, cs.cs, cs.v], cs.to_a)
  end

  def cs4(a, b)
    c0, s0 = Math.cos(a), Math.sin(a)
    c1, s1 = Math.cos(b), Math.sin(b)
    Matrix[[c0, 0, s0, 0], [0, c1, 0, s1], [-s0, 0, c0, 0], [0, -s1, 0, c1]]
  end

  def test_known_angles
    [[0.3, 1.0], [0.0, 0.4], [1e-12, 0.4], [Math::PI / 2 - 1e-12, 0.4],
     [Math::PI / 2, 0.4], [0.3, 0.3], [0.0, 0.0], [Math::PI / 2, Math::PI / 2],
     [1e-12, Math::PI / 2 - 1e-12]].each_with_index do |(a, b), idx|
      d = cs4(a, b)
      u0 = block_diag(random_orthogonal(2, idx), random_orthogonal(2, idx + 100))
      v0 = block_diag(random_orthogonal(2, idx + 200), random_orthogonal(2, idx + 300))
      m = u0 * d * v0.t
      cs = m.cs_decomposition(2, 2)

      assert_matrix_close(m, cs.u * cs.cs * cs.v.t)
      lo, hi = [a, b].minmax
      assert_in_delta(lo, cs.theta[0], 1e-8)
      assert_in_delta(hi, cs.theta[1], 1e-8)
      # square partition: all four blocks of cs are exactly diagonal
      [cs.cs.minor(0...2, 0...2), cs.cs.minor(0...2, 2...4),
       cs.cs.minor(2...4, 0...2), cs.cs.minor(2...4, 2...4)].each do |block|
        assert(block.each_with_index.all? { |x, i, j| i == j || x == 0 },
               "expected a diagonal block, got #{block}")
      end
    end
  end

  def test_documented_layout_unequal_partition
    # 7x7 example from the documentation of CSDecomposition, with p = 2, q = 4
    c0, s0 = Math.cos(0.3), Math.sin(0.3)
    c1, s1 = Math.cos(1.0), Math.sin(1.0)
    d = Matrix[
      [c0, 0,   0,  0,   0, s0, 0 ],
      [0,  c1,  0,  0,   0, 0,  s1],
      [0,  0,   0,  0,   1, 0,  0 ],
      [-s0, 0,  0,  0,   0, c0, 0 ],
      [0,  -s1, 0,  0,   0, 0,  c1],
      [0,  0,  -1,  0,   0, 0,  0 ],
      [0,  0,   0, -1,   0, 0,  0 ],
    ]
    u0 = block_diag(random_orthogonal(2, 11), random_orthogonal(5, 12))
    v0 = block_diag(random_orthogonal(4, 13), random_orthogonal(3, 14))
    m = u0 * d * v0.t
    cs = m.cs_decomposition(2, 4)

    assert_in_delta(0.3, cs.theta[0], EPS)
    assert_in_delta(1.0, cs.theta[1], EPS)
    assert_matrix_close(d, cs.cs)
    assert_matrix_close(m, cs.u * cs.cs * cs.v.t)
  end

  def test_identity
    cs = Matrix.identity(6).cs_decomposition(2, 4)
    assert_equal(Vector[0.0, 0.0], cs.theta)
    # the ones at positions (2,2), (3,3) and (4,4), (5,5) of the identity
    # fall into the Q21 and Q22 blocks, producing structural -1 and +1
    # entries in the cosine-sine matrix
    expected = Matrix[
      [1, 0,  0,  0, 0, 0],
      [0, 1,  0,  0, 0, 0],
      [0, 0,  0,  0, 1, 0],
      [0, 0,  0,  0, 0, 1],
      [0, 0, -1,  0, 0, 0],
      [0, 0,  0, -1, 0, 0],
    ]
    assert_equal(expected, cs.cs)
    assert_equal(Matrix.identity(6), cs.u * cs.cs * cs.v.t)
  end

  def test_permutation
    m = Matrix[[0, -1, 0], [1, 0, 0], [0, 0, -1]]
    csd = m.cs_decomposition(1, 1)
    u, cs, v = csd
    assert_in_delta(Math::PI / 2, csd.theta[0], EPS)
    assert_equal(m, (u * cs * v.t).map { |x| x.round(5) })
  end

  def test_errors
    assert_raise(Matrix::ErrDimensionMismatch) { Matrix[[1, 2], [3, 4], [5, 6]].cs_decomposition(1, 1) }
    assert_raise(ArgumentError) { Matrix.identity(4).cs_decomposition(0, 2) }
    assert_raise(ArgumentError) { Matrix.identity(4).cs_decomposition(4, 2) }
    assert_raise(ArgumentError) { Matrix.identity(4).cs_decomposition(2, 0) }
    assert_raise(ArgumentError) { Matrix.identity(4).cs_decomposition(2, 4) }
    assert_raise(TypeError) { Matrix.identity(4).cs_decomposition(1.5, 2) }
    assert_raise(TypeError) { Matrix.identity(4).cs_decomposition(2, nil) }
  end

  def test_alias
    m = random_orthogonal(4, 9)
    assert_equal(m.cs_decomposition(2, 2).theta, m.cos_sin_decomposition(2, 2).theta)
  end
end
