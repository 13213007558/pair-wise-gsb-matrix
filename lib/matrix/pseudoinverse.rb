# frozen_string_literal: false
class Matrix

  # Moore-Penrose pseudoinverse of a matrix.
  #
  # The pseudoinverse +A+⁺ of an <tt>m x n</tt> matrix +A+ is the unique
  # <tt>n x m</tt> matrix satisfying the four Moore-Penrose conditions:
  #
  #   A * A⁺ * A  == A
  #   A⁺ * A * A⁺ == A⁺
  #   (A * A⁺).adjoint == A * A⁺
  #   (A⁺ * A).adjoint == A⁺ * A
  #
  # It is computed from a singular value decomposition of +A+ obtained with
  # the one-sided Jacobi algorithm. This works for real and complex matrices
  # of any shape and, unlike methods based on inverting the normal equations
  # <tt>A*A x = A* b</tt>, it does not square the condition number of +A+.
  #
  # Singular values not greater than +rcond+ times the largest singular
  # value are treated as zero; the number of remaining ones is reported as
  # the numerical rank.
  #
  #   result = Matrix[[1, 2], [3, 4], [5, 6]].pseudoinverse
  #   result.matrix # => the 2x3 pseudoinverse
  #   result.rank   # => 2
  #
  class Pseudoinverse

    # Returns the pseudoinverse matrix. Its dimensions are the transpose of
    # the original matrix's dimensions.
    attr_reader :matrix

    # Returns the numerical rank of the matrix: the number of singular
    # values strictly greater than the cutoff threshold that was used.
    attr_reader :rank

    # Returns the computed singular values, in decreasing order.
    attr_reader :singular_values

    # Computes the pseudoinverse of the matrix +a+, which may be real or
    # complex and of any shape.
    #
    # +rcond+ is the relative cutoff for small singular values: those not
    # exceeding <tt>rcond * (largest singular value)</tt> are treated as
    # zero. It must be a real number in the interval <tt>[0, 1)</tt>.
    # The default is <tt>Float::EPSILON * max(row_count, column_count)</tt>.
    def initialize(a, rcond = nil)
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      case rcond
      when nil
        # default is used
      when Numeric
        raise TypeError, "rcond must be a real number, got #{rcond.inspect}" unless rcond.real?
        rcond = rcond.to_f
        unless 0.0 <= rcond && rcond < 1.0
          raise ArgumentError, "rcond must be in the interval [0, 1), got #{rcond}"
        end
      else
        raise TypeError, "rcond must be a real number, got #{rcond.class}"
      end

      row_count = a.row_count
      column_count = a.column_count

      if row_count == 0 || column_count == 0
        @matrix = Matrix.zero(column_count, row_count)
        @rank = 0
        @singular_values = [].freeze
        return
      end

      rcond ||= Float::EPSILON * [row_count, column_count].max

      # Work on columns; entries are Float for real matrices and Complex
      # (with Float components) otherwise.
      complex = a.any? {|e| e.is_a?(Complex) }
      columns = Array.new(column_count) do |j|
        Array.new(row_count) do |i|
          e = a[i, j]
          if complex
            e.is_a?(Complex) ? Complex(e.real.to_f, e.imaginary.to_f) : Complex(Float(e), 0.0)
          else
            Float(e)
          end
        end
      end
      # Accumulates the right singular vectors (as columns).
      v = Array.new(column_count) do |j|
        Array.new(column_count) {|i| i == j ? 1.0 : 0.0 }
      end

      # One-sided Jacobi SVD: orthogonalize the columns with Jacobi
      # rotations applied from the right, accumulating them in +v+.
      100.times do
        rotated = false
        0.upto(column_count - 2) do |p|
          (p + 1).upto(column_count - 1) do |q|
            x = columns[p]
            y = columns[q]
            alpha = beta = 0.0
            gamma = 0.0
            0.upto(row_count - 1) do |k|
              xk = x[k]
              yk = y[k]
              alpha += (xk * xk.conjugate).real
              beta += (yk * yk.conjugate).real
              gamma += xk.conjugate * yk
            end
            next if alpha == 0.0 || beta == 0.0
            gamma_abs = gamma.abs
            next if gamma_abs <= Float::EPSILON * Math.sqrt(alpha * beta)
            rotated = true
            if gamma.is_a?(Complex)
              # Phase rotation making the coupling coefficient real
              phase = gamma / gamma_abs
              y = columns[q] = y.map {|e| e / phase }
              v[q] = v[q].map {|e| e / phase }
              g = gamma_abs
            else
              g = gamma
            end
            # Jacobi rotation annihilating the off-diagonal element of the
            # 2x2 Gram matrix [[alpha, g], [g, beta]]
            tau = (beta - alpha) / (2.0 * g)
            t = (tau < 0 ? -1.0 : 1.0) / (tau.abs + Math.sqrt(1.0 + tau * tau))
            c = 1.0 / Math.sqrt(1.0 + t * t)
            s = t * c
            x = columns[p]
            0.upto(row_count - 1) do |k|
              xk = x[k]
              yk = y[k]
              x[k] = c * xk - s * yk
              y[k] = s * xk + c * yk
            end
            vp = v[p]
            vq = v[q]
            0.upto(column_count - 1) do |k|
              ek = vp[k]
              fk = vq[k]
              vp[k] = c * ek - s * fk
              vq[k] = s * ek + c * fk
            end
          end
        end
        break unless rotated
      end

      sigmas = columns.map do |col|
        Math.sqrt(col.inject(0.0) {|sum, e| sum + (e * e.conjugate).real })
      end
      order = (0...column_count).sort_by {|k| -sigmas[k] }

      cutoff = rcond * sigmas[order[0]]
      rank = order.count {|k| sigmas[k] > cutoff }

      # A⁺ = sum of (1/sigma_k) * v_k * u_k* over the kept singular values,
      # where u_k is the normalized k-th orthogonalized column.
      rows = Array.new(column_count) { Array.new(row_count, 0.0) }
      rank.times do |i|
        k = order[i]
        inv_sigma = 1.0 / sigmas[k]
        u = columns[k].map {|e| e * inv_sigma }
        vk = v[k]
        0.upto(column_count - 1) do |row|
          factor = vk[row] * inv_sigma
          next if factor == 0.0
          target = rows[row]
          0.upto(row_count - 1) do |j|
            target[j] += factor * u[j].conjugate
          end
        end
      end

      @matrix = Matrix.send(:new, rows, row_count)
      @rank = rank
      @singular_values = order.map {|k| sigmas[k] }
                              .first([row_count, column_count].min).freeze
    end
  end
end
