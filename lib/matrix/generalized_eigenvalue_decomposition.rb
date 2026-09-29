# frozen_string_literal: false
class Matrix
  # Generalized symmetric-definite eigenvalue decomposition.
  #
  # Solves +A v = lambda B v+ where +A+ is real symmetric and +B+ is real
  # symmetric positive definite. If +B = L L.t+ is the Cholesky factorization,
  # the problem is transformed to the symmetric eigenvalue problem
  # +C q = lambda q+ with +C = L.inverse * A * L.inverse.transpose+ and
  # +v = L.inverse.transpose * q+. The returned vectors satisfy
  # +V.transpose * B * V = I+.
  class GeneralizedEigenvalueDecomposition
    # Returns the eigenvector matrix +V+, normalized using the +B+ inner product.
    attr_reader :v

    # Returns the diagonal eigenvalue matrix +D+.
    attr_reader :d

    # Returns the inverse eigenvector matrix +V.transpose * B+.
    attr_reader :v_inv

    # Returns the eigenvalues in ascending order.
    attr_reader :eigenvalues

    # Returns residual and numerical-condition information.
    attr_reader :residuals

    alias_method :eigenvector_matrix, :v
    alias_method :eigenvalue_matrix, :d
    alias_method :eigenvector_matrix_inv, :v_inv

    def initialize(a, b)
      validate_matrices(a, b)

      @size = a.row_count
      if @size.zero?
        @a = a
        @b = b
        @v = Matrix.empty
        @d = Matrix.empty
        @v_inv = Matrix.empty
        @eigenvalues = []
        @residuals = {
          residual_matrix: Matrix.empty,
          residual_norms: [],
          relative_residual_norms: [],
          max_residual_norm: 0.0,
          max_relative_residual: 0.0,
          mass_matrix_residual: Matrix.empty,
          mass_residual_norm: 0.0,
          b_condition_number: 1.0
        }
        return
      end

      a_values = finite_float_matrix(a, "A")
      b_values = finite_float_matrix(b, "B")
      l = cholesky_factor(b_values)
      c = transformed_matrix(a_values, l)

      decomposition = EigenvalueDecomposition.new(Matrix.send(:new, c, @size))
      @eigenvalues = decomposition.eigenvalues.dup
      q = decomposition.eigenvector_matrix.to_a
      v_values = generalized_vectors(q, l)
      ensure_finite_computation(@eigenvalues, v_values)

      @a = Matrix.send(:new, a_values, @size)
      @b = Matrix.send(:new, b_values, @size)
      @v = Matrix.send(:new, v_values, @size)
      @d = Matrix.diagonal(*@eigenvalues)
      @v_inv = @v.transpose * @b
      @eigenvectors = nil
      @residuals = calculate_residuals
      ensure_finite_residuals
    end

    # Returns an array of the eigenvectors.
    def eigenvectors
      @eigenvectors ||= @v.column_vectors
    end

    # Returns [eigenvector_matrix, eigenvalue_matrix, eigenvector_matrix_inv].
    def to_ary
      [v, d, v_inv]
    end
    alias_method :to_a, :to_ary

    private

    def validate_matrices(a, b)
      unless a.is_a?(Matrix) && b.is_a?(Matrix)
        raise TypeError, "Expected A and B to be Matrix objects"
      end

      unless a.square? && b.square?
        raise Matrix::ErrDimensionMismatch, "A and B must be square matrices"
      end

      unless a.row_count == b.row_count
        raise Matrix::ErrDimensionMismatch, "A and B must have the same dimensions"
      end

      unless numeric_matrix?(a) && numeric_matrix?(b)
        raise TypeError, "A and B must contain numbers"
      end

      unless a.real? && b.real?
        raise TypeError, "A and B must contain real numbers"
      end

      unless finite_matrix?(a) && finite_matrix?(b)
        raise ArgumentError, "A and B must contain finite numbers"
      end

      unless a.symmetric?
        raise ArgumentError, "A must be symmetric"
      end

      unless b.symmetric?
        raise ArgumentError, "B must be symmetric"
      end
    end

    def finite_matrix?(matrix)
      matrix.all? {|entry| entry.is_a?(Numeric) && entry.real? && entry.finite? }
    end

    def numeric_matrix?(matrix)
      matrix.all? {|entry| entry.is_a?(Numeric) }
    end

    def finite_float_matrix(matrix, name)
      matrix.to_a.map do |row|
        row.map do |entry|
          value = Float(entry)
          unless value.finite?
            raise ArgumentError, "#{name} must contain finite numbers"
          end
          value
        end
      end
    rescue ArgumentError, TypeError
      raise if $!.is_a?(ArgumentError) && $!.message =~ /must contain finite numbers/
      raise TypeError, "#{name} must contain finite real numbers"
    end

    def cholesky_factor(values)
      factor = Array.new(@size) { Array.new(@size, 0.0) }

      @size.times do |i|
        @size.times do |j|
          next if j > i
          sum = values[i][j]
          j.times do |k|
            sum -= factor[i][k] * factor[j][k]
          end

          if i == j
            unless sum > 0.0 && sum.finite?
              raise Matrix::ErrNotPositiveDefinite, "B must be positive definite"
            end
            factor[i][i] = Math.sqrt(sum)
          else
            factor[i][j] = sum / factor[j][j]
          end
        end
      end

      factor
    end

    def transformed_matrix(a_values, l)
      y = Array.new(@size) { Array.new(@size, 0.0) }

      @size.times do |column|
        @size.times do |row|
          sum = a_values[row][column]
          row.times do |k|
            sum -= l[row][k] * y[k][column]
          end
          y[row][column] = sum / l[row][row]
        end
      end

      c = Array.new(@size) { Array.new(@size, 0.0) }

      @size.times do |row|
        x = Array.new(@size, 0.0)
        @size.times do |i|
          sum = y[row][i]
          i.times do |k|
            sum -= l[i][k] * x[k]
          end
          x[i] = sum / l[i][i]
        end
        @size.times {|column| c[row][column] = x[column] }
      end

      @size.times do |i|
        (i + 1).upto(@size - 1) do |j|
          symmetric_value = (c[i][j] + c[j][i]) / 2.0
          c[i][j] = c[j][i] = symmetric_value
        end
      end

      c
    end

    def generalized_vectors(q, l)
      vectors = Array.new(@size) { Array.new(@size, 0.0) }

      @size.times do |column|
        (@size - 1).downto(0) do |row|
          sum = q[row][column]
          (row + 1).upto(@size - 1) do |k|
            sum -= l[k][row] * vectors[k][column]
          end
          vectors[row][column] = sum / l[row][row]
        end
      end

      vectors
    end

    def ensure_finite_computation(values, vectors)
      unless values.all? {|value| value.real? && value.finite? } &&
             vectors.all? {|row| row.all? {|entry| entry.finite? } }
        raise Matrix::ErrNotRegular, "Generalized eigensystem computation is not numerically stable"
      end
    end

    def ensure_finite_residuals
      unless @residuals[:residual_matrix].all?(&:finite?) &&
             @residuals[:mass_matrix_residual].all?(&:finite?) &&
             @residuals[:residual_norms].all?(&:finite?) &&
             @residuals[:relative_residual_norms].all?(&:finite?)
        raise Matrix::ErrNotRegular, "Generalized eigensystem computation is not numerically stable"
      end
    end

    def calculate_residuals
      residual_matrix = @a * @v - @b * @v * @d
      mass_residual = @v.transpose * @b * @v - Matrix.identity(@size)
      a_norm = frobenius_norm(@a)
      b_norm = frobenius_norm(@b)

      residual_norms = eigenvectors.each_with_index.map do |vector, i|
        (residual_matrix.column(i)).magnitude
      end
      relative_residual_norms = eigenvectors.each_with_index.map do |vector, i|
        denominator = (a_norm + @eigenvalues[i].abs * b_norm) * vector.magnitude
        denominator = 1.0 if denominator.zero?
        residual_norms[i] / denominator
      end

      {
        residual_matrix: residual_matrix,
        residual_norms: residual_norms,
        relative_residual_norms: relative_residual_norms,
        max_residual_norm: residual_norms.max.to_f,
        max_relative_residual: relative_residual_norms.max.to_f,
        mass_matrix_residual: mass_residual,
        mass_residual_norm: frobenius_norm(mass_residual),
        b_condition_number: b_condition_number
      }
    end

    def b_condition_number
      return 1.0 if @size.zero?

      values = @b.eigensystem.eigenvalues
      minimum = values.min
      minimum > 0 ? values.max / minimum : Float::INFINITY
    end

    def frobenius_norm(matrix)
      Math.sqrt(matrix.inject(0.0) {|sum, entry| sum + entry.abs2 })
    end
  end
end
