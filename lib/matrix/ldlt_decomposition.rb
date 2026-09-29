# frozen_string_literal: false
class Matrix

  #
  # Bunch-Kaufman LDLt decomposition of a real symmetric matrix.
  #
  # For a symmetric n-by-n matrix A, the LDLt decomposition is a unit lower
  # triangular matrix L, a block diagonal matrix D (with 1-by-1 and 2-by-2
  # diagonal blocks), and a permutation matrix P so that
  #
  #   L*D*L.t == P*A*P.t
  #
  # or, equivalently,
  #
  #   A == P.t*L*D*L.t*P
  #
  # Symmetric pivoting (with 2-by-2 blocks when no suitable 1-by-1 pivot
  # exists) makes the decomposition exist for every symmetric matrix,
  # including indefinite ones and ones with zeros on the diagonal such as
  # Matrix[[0, 1], [1, 0]], so the constructor will never fail.
  #
  # The primary uses of the LDLt decomposition are the solution of symmetric
  # systems of linear equations (with one or many right hand sides) and the
  # computation of the inertia of A (the counts of positive, negative and
  # zero eigenvalues), which by Sylvester's law of inertia equals the
  # inertia of D.
  #
  class LDLTDecomposition

    include Matrix::ConversionHelper

    # Bunch-Kaufman pivoting constant: (1 + sqrt(17)) / 8
    ALPHA = 0.6403882032022076

    # Default relative tolerance (w.r.t. the largest absolute entry of the
    # matrix) below which a pivot is considered numerically zero.
    DEFAULT_TOLERANCE = Math.sqrt(Float::EPSILON)

    # Returns the unit lower triangular factor +L+

    def l
      Matrix.send :new, @l.map(&:dup), @size
    end

    # Returns the block diagonal factor +D+, whose diagonal blocks are
    # 1-by-1 or 2-by-2

    def d
      rows = Array.new(@size) { Array.new(@size, 0) }
      @size.times { |i| rows[i][i] = @d_diag[i] }
      each_block do |k, block_size|
        next if block_size == 1
        rows[k][k + 1] = rows[k + 1][k] = @d_off[k]
      end
      Matrix.send :new, rows, @size
    end

    # Returns the permutation matrix +P+, such that
    # <tt>L*D*L.t == P*A*P.t</tt>

    def p
      rows = Array.new(@size) { Array.new(@size, 0) }
      @pivots.each_with_index { |pivot, i| rows[i][pivot] = 1 }
      Matrix.send :new, rows, @size
    end

    # Returns +L+, +D+, +P+ in an array

    def to_ary
      [l, d, p]
    end
    alias_method :to_a, :to_ary

    # Returns the pivoting indices: row/column +i+ of the permuted matrix
    # <tt>P*A*P.t</tt> corresponds to row/column <tt>pivots[i]</tt> of +A+.

    attr_reader :pivots

    # Returns +true+ if +D+, and hence +A+, is singular.
    #
    # With the default <tt>tolerance: 0</tt> only exactly zero pivots make
    # the matrix singular. A positive +tolerance+ is interpreted relatively
    # to the largest absolute entry of +A+, so that numerically tiny pivots
    # also count as zero.

    def singular?(tolerance: 0)
      threshold = tolerance * @scale
      each_block do |k, block_size|
        if block_size == 1
          return true if @d_diag[k].abs <= threshold
        else
          det = @d_diag[k] * @d_diag[k + 1] - @d_off[k] * @d_off[k]
          return true if det.abs <= threshold * threshold
        end
      end
      false
    end

    # Returns +true+ if +A+ is numerically singular, i.e. if some pivot is
    # zero up to a relative tolerance (see #singular?). +tolerance+ defaults
    # to DEFAULT_TOLERANCE.

    def nearly_singular?(tolerance = DEFAULT_TOLERANCE)
      singular?(tolerance: tolerance)
    end

    # Returns the inertia of +A+ as an array
    # <tt>[positive, negative, zero]</tt> with the counts of positive,
    # negative and zero eigenvalues.
    #
    # By Sylvester's law of inertia these are the counts for the block
    # diagonal +D+; the signs of a 2-by-2 block are derived from its
    # determinant and trace, never from its two diagonal entries separately.
    #
    # A positive +tolerance+ (relative to the largest absolute entry of +A+)
    # classifies numerically tiny eigenvalues as zero.

    def inertia(tolerance: 0)
      positive = negative = zero = 0
      threshold = tolerance * @scale
      each_block do |k, block_size|
        if block_size == 1
          value = @d_diag[k]
          if value > threshold
            positive += 1
          elsif value < -threshold
            negative += 1
          else
            zero += 1
          end
        else
          trace = @d_diag[k] + @d_diag[k + 1]
          det = @d_diag[k] * @d_diag[k + 1] - @d_off[k] * @d_off[k]
          det_threshold = threshold * threshold
          if det < -det_threshold
            positive += 1
            negative += 1
          elsif det > det_threshold
            trace > 0 ? positive += 2 : negative += 2
          elsif trace > threshold
            positive += 1
            zero += 1
          elsif trace < -threshold
            negative += 1
            zero += 1
          else
            zero += 2
          end
        end
      end
      [positive, negative, zero]
    end

    # Returns +m+ so that <tt>A*m = b</tt>,
    # or equivalently so that <tt>L*D*L.t*m = P*b*... </tt> with
    # <tt>A == P.t*L*D*L.t*P</tt>.
    # +b+ can be a Matrix (each column is a right hand side) or a Vector.

    def solve b
      if (singular?)
        raise Matrix::ErrNotRegular, "Matrix is singular."
      end
      if b.is_a? Matrix
        if (b.row_count != @size)
          raise Matrix::ErrDimensionMismatch
        end

        # Copy right hand side with pivoting
        nx = b.column_count
        m = @pivots.map { |row| b.row(row).to_a }
        solve_permuted!(m, nx)

        rows = Array.new(@size)
        @pivots.each_with_index { |pivot, i| rows[pivot] = m[i] }
        Matrix.send :new, rows, nx
      else # same algorithm, specialized for simpler case of a vector
        b = convert_to_array(b)
        if (b.size != @size)
          raise Matrix::ErrDimensionMismatch
        end

        # Copy right hand side with pivoting
        m = b.values_at(*@pivots).map { |value| [value] }
        solve_permuted!(m, 1)

        values = Array.new(@size)
        @pivots.each_with_index { |pivot, i| values[pivot] = m[i][0] }
        Vector.elements(values, false)
      end
    end

    def initialize a
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      raise Matrix::ErrDimensionMismatch, "LDLt decomposition requires a square matrix" unless a.square?
      raise Matrix::ErrOperationNotDefined, "LDLt decomposition requires a symmetric matrix" unless a.symmetric?

      @size = a.row_count
      @a = a.to_a
      @scale = @a.flatten(1).map(&:abs).max || 0
      @l = Array.new(@size) { |i| Array.new(@size, 0).tap { |row| row[i] = 1 } }
      @d_diag = Array.new(@size, 0)
      @d_off = Array.new(@size, 0)
      @block_size = Array.new(@size, 1)
      @pivots = Array.new(@size) { |i| i }

      k = 0
      while k < @size
        # lambda = largest subdiagonal magnitude in column k, at row r
        lambda = 0
        r = k
        (k + 1).upto(@size - 1) do |i|
          magnitude = @a[i][k].abs
          if magnitude > lambda
            lambda = magnitude
            r = i
          end
        end

        if lambda == 0 || @a[k][k].abs >= ALPHA * lambda
          pivot_1x1(k)
          k += 1
          next
        end

        # sigma = largest off-diagonal magnitude in row/column r of the
        # trailing submatrix
        sigma = 0
        k.upto(@size - 1) do |j|
          next if j == r
          magnitude = @a[r][j].abs
          sigma = magnitude if magnitude > sigma
        end

        if @a[k][k].abs * sigma >= ALPHA * lambda * lambda
          pivot_1x1(k)
          k += 1
        elsif @a[r][r].abs >= ALPHA * sigma
          symmetric_swap(k, r, k)
          pivot_1x1(k)
          k += 1
        else
          symmetric_swap(k + 1, r, k)
          pivot_2x2(k)
          k += 2
        end
      end
    end

    private

    def each_block # :nodoc:
      k = 0
      while k < @size
        block_size = @block_size[k]
        yield k, block_size
        k += block_size
      end
    end

    # Swaps rows/columns +i+ and +j+ of the trailing submatrix, recording
    # the permutation; +computed+ is the number of finalized columns of +L+.
    def symmetric_swap(i, j, computed) # :nodoc:
      return if i == j
      @a[i], @a[j] = @a[j], @a[i]
      @size.times do |row|
        @a[row][i], @a[row][j] = @a[row][j], @a[row][i]
      end
      computed.times do |col|
        @l[i][col], @l[j][col] = @l[j][col], @l[i][col]
      end
      @pivots[i], @pivots[j] = @pivots[j], @pivots[i]
    end

    def pivot_1x1(k) # :nodoc:
      pivot = @a[k][k]
      @d_diag[k] = pivot
      return if pivot == 0
      (k + 1).upto(@size - 1) do |i|
        @l[i][k] = @a[i][k].quo(pivot)
      end
      (k + 1).upto(@size - 1) do |i|
        lik = @l[i][k]
        next if lik == 0
        (k + 1).upto(@size - 1) do |j|
          @a[i][j] -= lik * @a[k][j]
        end
      end
    end

    def pivot_2x2(k) # :nodoc:
      e11 = @a[k][k]
      e12 = @a[k][k + 1]
      e22 = @a[k + 1][k + 1]
      det = e11 * e22 - e12 * e12
      @block_size[k] = 2
      @block_size[k + 1] = 0
      @d_diag[k] = e11
      @d_diag[k + 1] = e22
      @d_off[k] = e12
      return if det == 0
      (k + 2).upto(@size - 1) do |i|
        b1 = @a[i][k]
        b2 = @a[i][k + 1]
        @l[i][k] = (b1 * e22 - b2 * e12).quo(det)
        @l[i][k + 1] = (b2 * e11 - b1 * e12).quo(det)
      end
      (k + 2).upto(@size - 1) do |i|
        li1 = @l[i][k]
        li2 = @l[i][k + 1]
        next if li1 == 0 && li2 == 0
        (k + 2).upto(@size - 1) do |j|
          @a[i][j] -= li1 * @a[k][j] + li2 * @a[k + 1][j]
        end
      end
    end

    # Solves <tt>L*D*L.t * m = rhs</tt> in place, where +m+ is an array of
    # @size rows of width +nx+, already permuted by @pivots.
    def solve_permuted!(m, nx) # :nodoc:
      # Solve L*y = m
      @size.times do |k|
        (k + 1).upto(@size - 1) do |i|
          lik = @l[i][k]
          next if lik == 0
          nx.times do |j|
            m[i][j] -= lik * m[k][j]
          end
        end
      end
      # Solve D*z = y
      each_block do |k, block_size|
        if block_size == 1
          pivot = @d_diag[k]
          nx.times do |j|
            m[k][j] = m[k][j].quo(pivot)
          end
        else
          e11 = @d_diag[k]
          e12 = @d_off[k]
          e22 = @d_diag[k + 1]
          det = e11 * e22 - e12 * e12
          nx.times do |j|
            y1 = m[k][j]
            y2 = m[k + 1][j]
            m[k][j] = (e22 * y1 - e12 * y2).quo(det)
            m[k + 1][j] = (e11 * y2 - e12 * y1).quo(det)
          end
        end
      end
      # Solve L.t*m = z
      (@size - 1).downto(0) do |k|
        (k + 1).upto(@size - 1) do |i|
          lik = @l[i][k]
          next if lik == 0
          nx.times do |j|
            m[k][j] -= lik * m[i][j]
          end
        end
      end
    end
  end
end
