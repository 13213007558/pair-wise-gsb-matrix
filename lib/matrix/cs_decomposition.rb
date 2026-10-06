# frozen_string_literal: false
class Matrix

  # Cosine-sine decomposition of a real orthogonal matrix.
  #
  # For an orthogonal n-by-n matrix +Q+ and a partition of its rows and
  # columns at positions +p+ and +q+ (with 0 < p < n and 0 < q < n):
  #
  #     Q = [ Q11 Q12 ]    Q11: p x q,      Q12: p x (n-q)
  #         [ Q21 Q22 ]    Q21: (n-p) x q,  Q22: (n-p) x (n-q)
  #
  # the cosine-sine decomposition (CSD) factorizes +Q+ as
  #
  #     Q = U * D * V.t
  #
  # where +U+ and +V+ are block-diagonal orthogonal matrices
  #
  #     U = [ U1  0  ]    V = [ V1  0  ]
  #         [ 0  U2 ]        [ 0  V2 ]
  #
  # with U1: p x p, U2: (n-p) x (n-p), V1: q x q, V2: (n-q) x (n-q),
  # and +D+ is an n-by-n matrix, blocked exactly like +Q+, whose four
  # blocks are (zero-padded) diagonal matrices:
  #
  #     D = [  C  S ]     C = diag(cos(theta_i))
  #         [ -S  C ]     S = diag(sin(theta_i))
  #
  # The angles 0 <= theta_0 <= ... <= theta_{k-1} <= PI/2, with
  # k = min(p, q), are the principal angles of the partition: cos(theta_i)
  # are the singular values of Q11. All four blocks share the single pair
  # of bases +U+ and +V+, so the decomposition stays consistent when the
  # angles are varied continuously (e.g. to interpolate between subspace
  # rotations, rebuild +D+ from new angles and keep +U+ and +V+ fixed).
  #
  # When the blocks are rectangular, +D+ also contains identity blocks
  # (from the directions where the partition is degenerate) and explicit
  # zero blocks; they are part of the returned #cs matrix, so
  # <tt>u * cs * v.t</tt> always reconstructs the whole original matrix,
  # including angles that are (nearly) 0 or PI/2 or repeated.
  #
  # == Example with an unequal partition
  #
  # For a 7 x 7 orthogonal matrix partitioned with p = 2 and q = 4:
  #
  #     Q = [ Q11 Q12 ]    Q11: 2 x 4,  Q12: 2 x 3
  #         [ Q21 Q22 ]    Q21: 5 x 4,  Q22: 5 x 3
  #
  # the returned factors have the following sizes:
  #
  #     u:  7 x 7, block diagonal of u1 (2 x 2) and u2 (5 x 5)
  #     v:  7 x 7, block diagonal of v1 (4 x 4) and v2 (3 x 3)
  #     cs: 7 x 7, blocked like Q
  #     theta: k = min(2, 4) = 2 angles
  #
  # and +cs+ has the layout (ci = cos(theta_i), si = sin(theta_i)):
  #
  #         [ c0  0   0   0  | 0   s0  0  ]
  #         [ 0   c1  0   0  | 0   0   s1 ]
  #         [ 0   0   0   0  | 1   0   0  ]
  #     D = [-s0  0   0   0  | 0   c0  0  ]
  #         [ 0  -s1  0   0  | 0   0   c1 ]
  #         [ 0   0  -1   0  | 0   0   0  ]
  #         [ 0   0   0  -1  | 0   0   0  ]
  #
  # The top-left 2 x 4 block is diag(c0, c1) padded with zero columns,
  # the top-right 2 x 3 block is diag(s0, s1) padded with a zero column,
  # the bottom-left 5 x 4 block is -diag(s0, s1) extended with a -1
  # identity part, and the bottom-right 5 x 3 block is diag(c0, c1)
  # extended with a +1 identity part and zero rows.
  #
  # == Example
  #
  #   q = Matrix[[0, -1, 0], [1, 0, 0], [0, 0, -1]]  # orthogonal
  #   csd = q.cs_decomposition(1, 1)
  #   csd.theta           # => Vector[1.5707963267948966]
  #   u, cs, v = csd
  #   (u * cs * v.t).round(5) == q   # => true
  #
  # Note that +Q+ must be orthogonal; no check is performed and the
  # result is meaningless otherwise.
  #
  class CSDecomposition
    # Returns the n-by-n block-diagonal orthogonal matrix +U+.
    attr_reader :u
    # Returns the n-by-n block-diagonal orthogonal matrix +V+.
    attr_reader :v
    # Returns the n-by-n cosine-sine matrix +D+ such that
    # <tt>Q == u * cs * v.t</tt> (up to rounding).
    attr_reader :cs
    # Returns the principal angles, a +Vector+ of min(p, q) angles in
    # [0, PI/2], sorted in ascending order.
    attr_reader :theta
    # Returns the +U1+ factor (p x p).
    attr_reader :u1
    # Returns the +U2+ factor ((n-p) x (n-p)).
    attr_reader :u2
    # Returns the +V1+ factor (q x q).
    attr_reader :v1
    # Returns the +V2+ factor ((n-q) x (n-q)).
    attr_reader :v2
    # Returns a +Vector+ with cos(theta_i) for each principal angle.
    attr_reader :cosines
    # Returns a +Vector+ with sin(theta_i) for each principal angle.
    attr_reader :sines

    # Computes the cosine-sine decomposition of the orthogonal matrix
    # +matrix+, partitioned after row +p+ and column +q+.
    def initialize(matrix, p, q)
      raise TypeError, "Expected Matrix but got #{matrix.class}" unless matrix.is_a?(Matrix)
      raise Matrix::ErrDimensionMismatch, "CS decomposition requires a square matrix" unless matrix.square?
      n = matrix.row_count
      [[:p, p], [:q, q]].each do |name, split|
        raise TypeError, "#{name} must be an Integer, got #{split.class}" unless split.is_a?(Integer)
        raise ArgumentError, "#{name} (#{split}) must be between 1 and #{n - 1}" unless 1 <= split && split < n
      end

      matrix = matrix.map(&:to_f)
      p1, p2 = p, n - p
      q1, q2 = q, n - q
      k = [p1, q1].min
      r = [p1, p2, q1, q2].min

      q11 = matrix.minor(0...p1, 0...q1)
      q12 = matrix.minor(0...p1, q1...n)
      q21 = matrix.minor(p1...n, 0...q1)
      q22 = matrix.minor(p1...n, q1...n)

      # V1 comes from the eigendecomposition of Q11' * Q11, with
      # eigenvectors ordered by decreasing eigenvalue (increasing angle).
      # The matrix is symmetrized first so the symmetric eigensolver is used.
      gram = q11.t * q11
      gram = (gram + gram.t) / 2
      eig = gram.eigensystem
      values = eig.eigenvalues
      vectors = eig.eigenvector_matrix
      order = (0...q1).sort_by { |i| [-values[i], i] }
      v1 = Matrix.columns(order.map { |i| vectors.column(i) })

      # The cosines and sines are computed directly as column norms, so
      # they stay accurate for angles close to 0 or PI/2 (no cancellation
      # of the form sqrt(1 - x**2)).
      m = q11 * v1   # column norms are the cosines
      e = q21 * v1   # column norms are the sines
      cos = Array.new(q1) { |i| m.column(i).magnitude }
      sin = Array.new(q1) { |i| e.column(i).magnitude }

      tol = 32.0 * n * Float::EPSILON

      # U1: normalize the columns of M; complete where a cosine vanishes.
      u1_cols = Array.new(p1)
      u1_w = Array.new(p1, 1.0)
      k.times do |i|
        if cos[i] > tol
          u1_cols[i] = m.column(i) / cos[i]
          u1_w[i] = cos[i]
        end
      end
      complete_columns!(u1_cols, p1)
      u1 = Matrix.columns(orthonormalize(u1_cols, u1_w))

      f = q12.t * u1   # column norms are the sines, once more
      sin12 = Array.new(p1) { |j| f.column(j).magnitude }

      # Column layout of U2 and V2:
      #   [ orthogonal complement | r principal angles | structural ]
      # where the structural columns appear when q1 > p1 (for U2) or
      # p1 > q1 (for V2) and correspond to angles of exactly PI/2.
      n_struct_u2 = [q1 - p1, 0].max
      n_struct_v2 = [p1 - q1, 0].max
      o2 = q2 - r - n_struct_v2
      o2p = p2 - r - n_struct_u2

      v2_cols = Array.new(q2)
      v2_w = Array.new(q2, 1.0)
      u2_cols = Array.new(p2)
      u2_w = Array.new(p2, 1.0)
      r.times do |t|
        i = k - r + t
        if sin[i] > tol && sin12[i] > 0
          v2_cols[o2 + t] = f.column(i) / sin12[i]
          v2_w[o2 + t] = sin12[i]
        end
        if sin[i] > tol
          u2_cols[o2p + t] = -e.column(i) / sin[i]
          u2_w[o2p + t] = sin[i]
        end
      end
      n_struct_v2.times do |t|
        i = q1 + t
        if sin12[i] > tol
          v2_cols[o2 + r + t] = f.column(i) / sin12[i]
          v2_w[o2 + r + t] = sin12[i]
        end
      end
      n_struct_u2.times do |t|
        i = p1 + t
        if sin[i] > tol
          u2_cols[o2p + r + t] = -e.column(i) / sin[i]
          u2_w[o2p + r + t] = sin[i]
        end
      end

      # Complete V2 arbitrarily, then complete U2 by pairing each missing
      # column with the matching V2 column through Q22, so that the (2,2)
      # block of D holds an identity block there. This keeps the bases of
      # the four blocks mutually compatible.
      complete_columns!(v2_cols, q2)
      basis = u2_cols.compact
      p2.times do |pos|
        next if u2_cols[pos]
        col = nil
        if pos < q2
          cand = q22 * v2_cols[pos]
          2.times { basis.each { |b| cand -= b * b.inner_product(cand) } }
          col = cand / cand.magnitude if cand.magnitude > 0.5
        end
        unless col
          p2.times do |j|
            cand = Vector.basis(size: p2, index: j)
            2.times { basis.each { |b| cand -= b * b.inner_product(cand) } }
            if cand.magnitude > 0.5
              col = cand / cand.magnitude
              break
            end
          end
        end
        raise Matrix::ErrNotRegular, "cannot complete an orthogonal basis" unless col
        basis << (u2_cols[pos] = col)
      end

      v2 = Matrix.columns(orthonormalize(v2_cols, v2_w))
      u2 = Matrix.columns(orthonormalize(u2_cols, u2_w))

      @u1, @u2, @v1, @v2 = u1, u2, v1, v2
      @u = Matrix.hstack(Matrix.vstack(u1, Matrix.zero(p2, p1)),
                         Matrix.vstack(Matrix.zero(p1, p2), u2))
      @v = Matrix.hstack(Matrix.vstack(v1, Matrix.zero(q2, q1)),
                         Matrix.vstack(Matrix.zero(q1, q2), v2))

      # Clean up the tiny off-structure entries so that the zero and
      # identity blocks of D are exact.
      d = @u.t * matrix * @v
      clean = 8.0 * n * Float::EPSILON
      @cs = d.map do |x|
        if x.abs <= clean
          0.0
        elsif (x - 1.0).abs <= clean
          1.0
        elsif (x + 1.0).abs <= clean
          -1.0
        else
          x
        end
      end

      @theta = Vector.elements(Array.new(k) { |i| Math.atan2(sin[i], cos[i]) })
      @cosines = Vector.elements(cos.first(k))
      @sines = Vector.elements(sin.first(k))
    end

    # Returns [u, cs, v]
    def to_ary
      [u, cs, v]
    end
    alias_method :to_a, :to_ary

    # Orthonormalizes a list of column vectors with twice-repeated
    # Gram-Schmidt. Columns are processed by decreasing weight (the norm
    # of the column they were derived from), so that well-determined
    # directions are kept and ill-determined ones (from angles near 0 or
    # PI/2) are adjusted against them.
    private def orthonormalize(cols, weights)
      basis = []
      result = Array.new(cols.size)
      cols.each_with_index.sort_by { |_, i| [-weights[i], i] }.each do |col, i|
        2.times { basis.each { |b| col -= b * b.inner_product(col) } }
        basis << (result[i] = col / col.magnitude)
      end
      result
    end

    # Fills the nil entries of +cols+ with unit vectors orthogonal to all
    # the other columns.
    private def complete_columns!(cols, dim)
      basis = cols.compact
      cols.each_index do |i|
        next if cols[i]
        col = nil
        dim.times do |j|
          cand = Vector.basis(size: dim, index: j)
          2.times { basis.each { |b| cand -= b * b.inner_product(cand) } }
          if cand.magnitude > 0.5
            col = cand / cand.magnitude
            break
          end
        end
        raise Matrix::ErrNotRegular, "cannot complete an orthogonal basis" unless col
        basis << (cols[i] = col)
      end
      cols
    end
  end
end
