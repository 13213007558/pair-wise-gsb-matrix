# frozen_string_literal: false
class Matrix
  # Eigenvalues and eigenvectors of the real symmetric generalized problem
  # A*v = lambda*B*v, where B is symmetric and positive definite.
  #
  # The eigenvector matrix V satisfies V.t*B*V = I, and therefore
  # A*V = B*V*D and V.inv = V.t*B.
  class GeneralizedEigenvalueDecomposition
    attr_reader :eigenvalues, :residual_matrix, :residual_norms,
                :relative_residual_norms, :orthonormality_error,
                :b_condition_number

    def initialize(a, b)
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      raise TypeError, "Expected Matrix but got #{b.class}" unless b.is_a?(Matrix)

      unless a.square? && b.square? && a.row_count == b.row_count
        raise Matrix::ErrDimensionMismatch, "Matrices A and B must be square matrices of the same size"
      end

      @size = a.row_count
      if @size.zero?
        raise ArgumentError, "can't decompose an empty matrix"
      end

      raise ArgumentError, "matrix A must contain only finite real values" unless finite_real?(a)
      raise ArgumentError, "matrix B must contain only finite real values" unless finite_real?(b)
      raise ArgumentError, "matrix A must be symmetric" unless a.symmetric?
      raise ArgumentError, "matrix B must be symmetric" unless b.symmetric?

      @a = to_float_matrix(a)
      @b = to_float_matrix(b)
      @l = cholesky(@b)
      c = inverse_lower_transpose(@l)
      transformed = c.transpose * @a * c
      transformed = (transformed + transformed.transpose) * 0.5

      eigensystem = EigenvalueDecomposition.new(transformed)
      @eigenvalues = eigensystem.eigenvalues
      raw_vectors = c * eigensystem.v
      @eigenvector_matrix = b_orthonormalize(raw_vectors, @l)

      calculate_residuals
    end

    def eigenvector_matrix
      @eigenvector_matrix
    end
    alias_method :v, :eigenvector_matrix

    def eigenvector_matrix_inv
      @eigenvector_matrix.transpose * @b
    end
    alias_method :v_inv, :eigenvector_matrix_inv

    def eigenvectors
      @eigenvector_matrix.column_vectors
    end

    def eigenvalue_matrix
      Matrix.diagonal(*@eigenvalues)
    end
    alias_method :d, :eigenvalue_matrix

    def residuals
      {
        matrix: residual_matrix,
        norms: residual_norms,
        relative_residual_norms: relative_residual_norms,
        orthonormality_error: orthonormality_error,
        b_condition_number: b_condition_number
      }
    end

    def to_ary
      [v, d, v_inv]
    end
    alias_method :to_a, :to_ary

    private

    def finite_real?(matrix)
      matrix.all? do |entry|
        entry.is_a?(Numeric) && !entry.is_a?(Complex) &&
          entry.respond_to?(:finite?) && entry.finite?
      end
    end

    def to_float_matrix(matrix)
      rows = matrix.to_a.map do |row|
        row.map do |entry|
          value = entry.to_f
          raise ArgumentError, "matrix must contain finite Float values" unless value.finite?
          value
        end
      end
      Matrix.send(:new, rows)
    end

    def cholesky(matrix)
      rows = Array.new(@size) { Array.new(@size, 0.0) }

      @size.times do |i|
        @size.times do |j|
          next if i < j

          value = matrix[i, j]
          [i, j].min.times do |k|
            value -= rows[i][k] * rows[j][k]
          end

          if i == j
            unless value.finite? && value > 0.0
              raise Matrix::ErrNotRegular, "matrix B must be positive definite"
            end
            value = Math.sqrt(value)
            unless value.finite? && value > 0.0
              raise Matrix::ErrNotRegular, "matrix B is too ill-conditioned to factorize"
            end
          else
            value /= rows[j][j]
          end

          rows[i][j] = value
        end
      end

      Matrix.send(:new, rows)
    end

    def inverse_lower_transpose(lower)
      rows = Array.new(@size) { Array.new(@size, 0.0) }

      @size.times do |column|
        (@size - 1).downto(0) do |row|
          value = row == column ? 1.0 : 0.0
          (row + 1).upto(@size - 1) do |k|
            value -= lower[k, row] * rows[k][column]
          end
          rows[row][column] = value / lower[row, row]
        end
      end

      Matrix.send(:new, rows)
    end

    def b_orthonormalize(vectors, lower)
      euclidean_basis = []
      columns = []

      @size.times do |j|
        vector = vectors.column(j).to_a
        normalized = multiply_lower_transpose(lower, vector)

        # Modified Gram-Schmidt twice improves orthogonality for tightly
        # clustered eigenvalues and badly scaled positive definite matrices.
        2.times do
          j.times do |k|
            coefficient = 0.0
            @size.times { |i| coefficient += normalized[i] * euclidean_basis[k][i] }
            @size.times { |i| normalized[i] -= coefficient * euclidean_basis[k][i] }
          end
        end

        norm = Math.sqrt(normalized.sum { |entry| entry * entry })
        unless norm.finite? && norm > 0.0
          raise Matrix::ErrNotRegular, "matrix B is too ill-conditioned to normalize eigenvectors"
        end
        normalized.map! { |entry| entry / norm }

        euclidean_basis[j] = normalized
        columns[j] = solve_lower_transpose(lower, normalized)
      end

      Matrix.send(:new, columns.transpose)
    end

    def multiply_lower_transpose(lower, vector)
      Array.new(@size) do |row|
        value = 0.0
        row.upto(@size - 1) do |column|
          value += lower[column, row] * vector[column]
        end
        value
      end
    end

    def solve_lower_transpose(lower, vector)
      solution = Array.new(@size, 0.0)

      (@size - 1).downto(0) do |row|
        value = vector[row]
        (row + 1).upto(@size - 1) do |column|
          value -= lower[column, row] * solution[column]
        end
        solution[row] = value / lower[row, row]
      end

      solution
    end

    def calculate_residuals
      av = @a * @eigenvector_matrix
      bv = @b * @eigenvector_matrix
      @residual_matrix = av - bv * eigenvalue_matrix
      @residual_norms = @residual_matrix.column_vectors.map(&:norm)

      @relative_residual_norms = @eigenvalues.each_with_index.map do |value, i|
        denominator = av.column(i).norm + value.abs * bv.column(i).norm
        if denominator.zero?
          @residual_norms[i]
        else
          @residual_norms[i] / denominator
        end
      end

      gram = @eigenvector_matrix.transpose * @b * @eigenvector_matrix
      @orthonormality_error = frobenius_norm(gram - Matrix.identity(@size))

      b_eigensystem = EigenvalueDecomposition.new((@b + @b.transpose) * 0.5)
      b_eigenvalues = b_eigensystem.eigenvalues
      minimum = b_eigenvalues.first
      maximum = b_eigenvalues.last
      @b_condition_number = minimum > 0 ? maximum / minimum : Float::INFINITY

      unless @eigenvalues.all?(&:finite?) &&
             [@residual_norms, @relative_residual_norms].flatten.all?(&:finite?) &&
             @orthonormality_error.finite?
        raise Matrix::ErrNotRegular, "generalized eigensystem contains non-finite numerical results"
      end
    end

    def frobenius_norm(matrix)
      Math.sqrt(matrix.to_a.sum { |row| row.sum { |entry| entry * entry } })
    end
  end
end
