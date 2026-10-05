# frozen_string_literal: false

class Matrix
  class PrincipalLogarithm
    Result = Struct.new(:logarithm, :residual, :relative_residual, :iterations, keyword_init: true)

    class << self
      def compute(matrix, precision, diagnostics)
        new(matrix, precision, diagnostics).compute
      end
    end

    def initialize(matrix, precision, diagnostics)
      @matrix = matrix
      @precision = precision
      @diagnostics = diagnostics
      @iterations = 0
    end

    def compute
      validate

      decomposition = EigenvalueDecomposition.new(@matrix, false)
      validate_spectrum(decomposition.eigenvalues)

      t, z = complex_schur(decomposition.schur_matrix, decomposition.schur_vectors)
      f = parlett_log(t)

      logarithm = z * f * z.inverse
      imaginary = matrix_frobenius(logarithm.imaginary)
      if !imaginary.finite? || imaginary > @precision * [1.0, matrix_frobenius(logarithm.real)].max
        raise Matrix::ErrPrincipalLogarithm, "principal logarithm has a non-real residual of #{imaginary}"
      end

      logarithm = logarithm.real
      residual_matrix = schur_exponential_residual(logarithm)
      residual = matrix_frobenius(residual_matrix)
      relative = residual / [1.0, matrix_frobenius(@matrix)].max
      if relative > @precision
        raise Matrix::ErrPrincipalLogarithm,
              "unable to attain requested precision #{@precision}; relative residual #{relative}"
      end

      result = logarithm
      if @diagnostics
        Result.new(
          logarithm: logarithm,
          residual: residual,
          relative_residual: relative,
          iterations: @iterations
        )
      else
        result
      end
    end

    private

    def validate
      unless @matrix.is_a?(Matrix) && @matrix.square? && !@matrix.empty?
        raise Matrix::ErrPrincipalLogarithm, "principal logarithm requires a non-empty square Matrix"
      end
      unless @matrix.real?
        raise Matrix::ErrPrincipalLogarithm, "principal logarithm requires a real matrix"
      end
      unless @matrix.to_a.all? { |row| row.all? { |entry| entry.is_a?(Numeric) && entry.finite? } }
        raise Matrix::ErrPrincipalLogarithm, "principal logarithm requires finite numeric entries"
      end
      unless @precision.is_a?(Numeric) && @precision > 0 && @precision.finite?
        raise ArgumentError, "precision must be a positive finite number"
      end
    end

    def validate_spectrum(values)
      values.each do |value|
        if value.real?
          if value == 0
            raise Matrix::ErrPrincipalLogarithm, "zero eigenvalue is on the principal-logarithm branch cut"
          elsif value < 0
            raise Matrix::ErrPrincipalLogarithm, "negative real eigenvalue #{value} is on the principal-logarithm branch cut"
          end
        elsif value.imag == 0 && value.real <= 0
          raise Matrix::ErrPrincipalLogarithm, "eigenvalue #{value} is on the principal-logarithm branch cut"
        end
      end
    end

    def complex_schur(real_t, real_z)
      n = real_t.row_count
      t = Matrix.build(n) { |i, j| Complex(real_t[i, j], 0) }
      z = Matrix.build(n) { |i, j| Complex(real_z[i, j], 0) }
      i = 0

      while i < n
        if i + 1 < n && real_t[i + 1, i] != 0
          a = real_t[i, i]
          b = real_t[i, i + 1]
          c = real_t[i + 1, i]
          d = real_t[i + 1, i + 1]
          eigenvalue = decomposition_eigenvalue(a, b, c, d)
          beta = Math.sqrt(eigenvalue.imag.abs / c.abs)
          alpha = ((a - d) / 2.0) / beta

          v = Matrix[
            [Complex(1.0, 0.0), Complex(1.0, 0.0)],
            [Complex(alpha, -1.0), Complex(alpha, 1.0)]
          ]
          apply_complex_block_similarity!(t, z, i, v)
          i += 2
        else
          i += 1
        end
      end

      [cleanup_triangular(t), z]
    end

    def decomposition_eigenvalue(a, b, c, d)
      discriminant = ((a + d) / 2.0)**2 - (a * d - b * c)
      imaginary = Math.sqrt([-discriminant, 0].max)
      Complex((a + d) / 2.0, imaginary)
    end

    def apply_complex_block_similarity!(t, z, offset, v)
      n = t.row_count
      m = v.row_count
      full = identity_complex(n)
      m.times do |i|
        m.times do |j|
          full[offset + i, offset + j] = v[i, j]
        end
      end
      transformed = full.inverse * t * full
      n.times do |i|
        n.times do |j|
          t[i, j] = transformed[i, j]
        end
      end
      rotated = z * full
      n.times do |i|
        n.times do |j|
          z[i, j] = rotated[i, j]
        end
      end
    end

    def cleanup_triangular(t)
      Matrix.build(t.row_count) do |i, j|
        i > j ? 0 : t[i, j]
      end
    end

    def parlett_log(t)
      n = t.row_count
      f = zero_complex(n)
      boundaries = cluster_boundaries(t)

      boundaries.each do |top, bottom|
        block = block_matrix(t, top, bottom)
        logged = log_triangular_block(block)
        (top..bottom).each do |i|
          (top..bottom).each do |j|
            f[i, j] = logged[i - top, j - top]
          end
        end
      end

      (1...boundaries.size).each do |right_index|
        right_top, right_bottom = boundaries[right_index]
        (right_index - 1).downto(0) do |left_index|
          left_top, left_bottom = boundaries[left_index]
          coupling = block_matrix(t, left_top, left_bottom, right_top, right_bottom)
          (left_index + 1...right_index).each do |middle_index|
            middle_top, middle_bottom = boundaries[middle_index]
            coupling += block_matrix(t, left_top, left_bottom, middle_top, middle_bottom) *
                        block_matrix(f, middle_top, middle_bottom, right_top, right_bottom) -
                        block_matrix(f, left_top, left_bottom, middle_top, middle_bottom) *
                        block_matrix(t, middle_top, middle_bottom, right_top, right_bottom)
          end
          left_f = block_matrix(f, left_top, left_bottom)
          right_f = block_matrix(f, right_top, right_bottom)
          rhs = coupling * right_f - left_f * coupling
          solution = solve_complex_sylvester(
            block_matrix(t, left_top, left_bottom),
            block_matrix(t, right_top, right_bottom),
            rhs
          )
          solution.row_count.times do |i|
            solution.column_count.times do |j|
              f[left_top + i, right_top + j] = solution[i, j]
            end
          end
        end
      end

      f
    end

    def cluster_boundaries(t)
      boundaries = []
      start = 0
      (1...t.row_count).each do |i|
        if (principal_scalar_log(t[i - 1, i - 1]) - principal_scalar_log(t[i, i])).abs >= 0.02
          boundaries << [start, i - 1]
          start = i
        end
      end
      boundaries << [start, t.row_count - 1]
      boundaries
    end

    def log_triangular_block(block)
      n = block.row_count
      if n > 2 && (block[0, 0] - block[n - 1, n - 1]).abs > 0
        return log_distinct_triangular_block(block)
      end
      center = block[0, 0]
      validate_scalar_branch(center)

      shifted = block - Matrix.scalar(n, center)
      result = Matrix.scalar(n, principal_scalar_log(center))
      power = identity_complex(n)
      500.times do |degree|
        power = power * shifted
        coefficient = ((-1.0) ** degree) / ((degree + 1) * center ** (degree + 1))
        term = coefficient * power
        result += term
        @iterations += 1
        if matrix_frobenius(term) <= @precision * [1.0, matrix_frobenius(result)].max
          return result
        end
      end

      raise Matrix::ErrPrincipalLogarithm, "Taylor series for a logarithm cluster did not converge"
    end

    def log_distinct_triangular_block(block)
      n = block.row_count
      result = Matrix.zero(n)
      n.times { |i| result[i, i] = principal_scalar_log(block[i, i]) }
      (1...n).each do |j|
        (j - 1).downto(0) do |i|
          sum = block[i, j]
          (i + 1...j).each do |k|
            sum += result[i, k] * block[k, j] - block[i, k] * result[k, j]
          end
          result[i, j] = sum / (result[j, j] - result[i, i])
        end
      end
      result
    end

    def validate_scalar_branch(value)
      return unless value.imag.abs < @precision * [1.0, value.real.abs].max && value.real <= 0

      raise Matrix::ErrPrincipalLogarithm, "numerical eigenvalue #{value} is on or too close to the branch cut"
    end

    def principal_scalar_log(value)
      Complex(Math.log(Math.hypot(value.real, value.imag)), Math.atan2(value.imag, value.real))
    end

    def schur_exponential_residual(logarithm)
      decomposition = EigenvalueDecomposition.new(logarithm, false)
      t = decomposition.schur_matrix.map(&:to_f)
      z = decomposition.schur_vectors.map(&:to_f)
      f = real_schur_exponential(t)
      z * f * z.transpose - @matrix
    rescue => error
      raise Matrix::ErrPrincipalLogarithm, "unable to compute an exponential residual: #{error.message}"
    end

    def real_schur_exponential(t)
      n = t.row_count
      blocks = exponential_blocks(t)
      f = Matrix.zero(n)
      blocks.each do |top, bottom|
        block = t.minor(top..bottom, top..bottom)
        exponent = if block.row_count > 1 && block[1, 0] == 0
          triangular_block_exponential(block)
        else
          block_exponential(block)
        end
        (top..bottom).each do |i|
          (top..bottom).each do |j|
            f[i, j] = exponent[i - top, j - top]
          end
        end
      end

      (1...blocks.size).each do |right_index|
        right_top, right_bottom = blocks[right_index]
        (right_index - 1).downto(0) do |left_index|
          left_top, left_bottom = blocks[left_index]
          coupling = block_matrix(t, left_top, left_bottom, right_top, right_bottom)
          middle = Matrix.zero(coupling.row_count, coupling.column_count)
          (left_index + 1...right_index).each do |middle_index|
            middle_top, middle_bottom = blocks[middle_index]
            middle += block_matrix(f, left_top, left_bottom, middle_top, middle_bottom) *
                       block_matrix(t, middle_top, middle_bottom, right_top, right_bottom) -
                       block_matrix(t, left_top, left_bottom, middle_top, middle_bottom) *
                       block_matrix(f, middle_top, middle_bottom, right_top, right_bottom)
          end
          left_t = block_matrix(t, left_top, left_bottom)
          left_f = block_matrix(f, left_top, left_bottom)
          right_t = block_matrix(t, right_top, right_bottom)
          right_f = block_matrix(f, right_top, right_bottom)
          coupling = left_f * coupling - coupling * right_f + middle
          solution = exponential_block_solution(
            left_t,
            right_t,
            coupling
          )
          solution.row_count.times do |i|
            solution.column_count.times do |j|
              f[left_top + i, right_top + j] = solution[i, j]
            end
          end
        end
      end

      f
    end

    def exponential_blocks(t)
      blocks = []
      i = 0
      while i < t.row_count
        if i + 1 < t.row_count && t[i + 1, i] != 0
          blocks << [i, i + 1]
          i += 2
        else
          start = i
          value = t[i, i]
          i += 1
          while i < t.row_count &&
                (i + 1 >= t.row_count || t[i + 1, i] == 0) &&
                (t[i, i] - value).abs <= 1e-12 * [1.0, value.abs].max
            i += 1
          end
          blocks << [start, i - 1]
        end
      end
      blocks
    end

    def exponential_block_solution(left_t, right_t, coupling)
      rows = left_t.row_count
      columns = right_t.column_count
      coefficients = Matrix.build(rows * columns) do |p, q|
        i = p / columns
        j = p % columns
        k = q / columns
        l = q % columns
        value = 0
        value += left_t[i, k] if j == l
        value -= right_t[l, j] if i == k
        value
      end
      vector = coefficients.lup.solve(Vector.elements(coupling.to_a.flatten))
      Matrix.build(rows, columns) { |i, j| vector[i * columns + j] }
    end

    def block_exponential(block)
      n = block.row_count
      scale = [0, Math.log2([1.0, matrix_frobenius(block)].max).ceil].max
      scaled = block / (2.0 ** scale)
      degree = 13
      numerator = Matrix.zero(n)
      denominator = Matrix.identity(n)
      power = Matrix.identity(n)
      coefficient = 1.0
      (1..degree).each do |k|
        coefficient *= Rational(degree - k + 1, k * (2 * degree - k + 1))
        power = power * scaled
        term = coefficient.to_f * power
        if k.odd?
          numerator += term
        else
          denominator += term
        end
      end
      result = (denominator - numerator).inverse * (denominator + numerator)
      scale.times { result = result * result }
      result
    end

    def triangular_block_exponential(block)
      n = block.row_count
      if n == 1 || (block[0, 0] - block[n - 1, n - 1]).abs > 1e-12 * [1.0, block[0, 0].abs].max
        result = Matrix.zero(n)
        n.times { |i| result[i, i] = Math.exp(block[i, i]) }
        (1...n).each do |j|
          (j - 1).downto(0) do |i|
            sum = block[i, j] * (result[j, j] - result[i, i])
            (i + 1...j).each do |k|
              sum += result[i, k] * block[k, j] - block[i, k] * result[k, j]
            end
            result[i, j] = sum / (block[j, j] - block[i, i])
          end
        end
        result
      else
        diagonal = block[0, 0]
        shifted = block - Matrix.scalar(n, diagonal)
        series = Matrix.identity(n)
        power = Matrix.identity(n)
        100.times do |degree|
          power = power * shifted / (degree + 1)
          series += power
          if matrix_frobenius(power) <= @precision * [1.0, matrix_frobenius(series)].max
            break
          end
        end
        Math.exp(diagonal) * series
      end
    end

    def block_matrix(matrix, top, bottom, column_top = top, column_bottom = bottom)
      rows = bottom - top + 1
      columns = column_bottom - column_top + 1
      Matrix.build(rows, columns) do |i, j|
        matrix[top + i, column_top + j]
      end
    end

    def solve_complex_sylvester(left, right, rhs)
      rows = left.row_count
      columns = right.column_count
      coefficients = Matrix.build(rows * columns) do |p, q|
        i = p / columns
        j = p % columns
        k = q / columns
        l = q % columns
        value = 0
        value -= left[i, k] if j == l
        value += right[l, j] if i == k
        value
      end
      vector = coefficients.lup.solve(Vector.elements(rhs.to_a.flatten))
      Matrix.build(rows, columns) { |i, j| vector[i * columns + j] }
    end

    def identity_complex(size)
      Matrix.build(size) { |i, j| Complex(i == j ? 1 : 0, 0) }
    end

    def zero_complex(size)
      Matrix.build(size) { Complex(0, 0) }
    end

    def matrix_frobenius(matrix)
      sum = matrix.row_count.times.sum do |i|
        matrix.column_count.times.sum do |j|
          value = matrix[i, j]
          value.is_a?(Complex) ? value.abs2 : value.abs2
        end
      end
      Math.sqrt(sum)
    end
  end
end
