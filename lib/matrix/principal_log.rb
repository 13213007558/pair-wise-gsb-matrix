# frozen_string_literal: false

class Matrix
  class PrincipalLog # :nodoc: all
    MAX_SQUARE_ROOTS = 128
    SERIES_THRESHOLD = 0.105
    SERIES_TERMS = 16
    EXPONENTIAL_TERMS = 24

    def initialize(matrix, precision, diagnostics)
      @matrix = matrix
      @precision = precision
      @diagnostics = diagnostics
      @size = matrix.row_count
      @epsilon = Float::EPSILON
    end

    def result
      validate
      return empty_result if @size.zero?

      matrix = @matrix.send(:new_matrix, @a)
      decomposition = matrix.eigensystem
      reject_branch_cut(decomposition.eigenvalues, matrix_norm1(matrix))
      schur_matrix = decomposition.schur_matrix
      schur_vectors = decomposition.schur_vectors
      schur_array = schur_matrix.to_a
      clear_real_splits(schur_array)
      schur_matrix = matrix.send(:new_matrix, schur_array)
      blocks = schur_blocks(schur_array)

      if @size == 2 && near_negative_complex_block?(schur_array, blocks)
        log_schur = schur_logarithm(schur_matrix, blocks)
        reconstructed = matrix_exponential(log_schur)
        residual_data = residual(matrix, schur_vectors, reconstructed, log_schur, 0)
        return final_result(residual_data) if residual_data[:relative_residual] <= @precision
        return failed_result(residual_data)
      end

      inverse_scaling_log(matrix, schur_matrix, schur_vectors)
    end

    private

    def validate
      unless @precision.is_a?(Numeric) && @precision.real? &&
             @precision >= @epsilon && @precision < 1.0
        raise ArgumentError, "precision must be a real number in [#{Float::EPSILON}, 1)"
      end
      @matrix.each do |entry|
        unless entry.is_a?(Numeric) && entry.real?
          raise TypeError, "principal logarithm requires a real matrix"
        end
      end
      @a = @matrix.to_a.map {|row| row.map(&:to_f)}
    end

    def empty_result
      result = @matrix.send(:new_matrix, @a)
      @diagnostics ? {
        matrix: result,
        residual: 0.0,
        relative_residual: 0.0,
        square_roots: 0,
        series_terms: 0
      } : result
    end

    def reject_branch_cut(eigenvalues, matrix_norm)
      tolerance = 100.0 * @epsilon * [1.0, matrix_norm].max
      eigenvalues.each do |eigenvalue|
        imaginary = eigenvalue.imaginary.abs
        real = eigenvalue.real
        next unless imaginary <= tolerance * [1.0, real.abs].max
        next if real > tolerance

        description = if real <= 0 && imaginary.zero?
          real.zero? ? "zero eigenvalue" : "negative real eigenvalue #{real}"
        else
          "eigenvalue #{eigenvalue.inspect} is numerically on the non-positive real axis"
        end
        raise ArgumentError,
          "principal matrix logarithm is not defined for spectra containing zero or the negative real axis: #{description}"
      end
    end

    def inverse_scaling_log(matrix, schur_matrix, schur_vectors)
      square_roots = 0
      loop do
        schur_array = schur_matrix.to_a
        clear_real_splits(schur_array)
        r_norm = array_norm1(subtract_identity(schur_array))
        if r_norm <= SERIES_THRESHOLD
          schur_matrix = matrix.send(:new_matrix, schur_array)
          r = schur_matrix - Matrix.identity(@size)
          log_series = logarithm_series(r)
          log_schur = log_series * (2.0 ** square_roots)
          reconstructed = reconstruct_exponential(log_series, square_roots)
          residual_data = residual(matrix, schur_vectors, reconstructed, log_schur, square_roots)
          return final_result(residual_data) if residual_data[:relative_residual] <= @precision
          return failed_result(residual_data) if square_roots >= MAX_SQUARE_ROOTS
        end

        schur_matrix = schur_square_root(schur_matrix)
        square_roots += 1
      end
    end

    def near_negative_complex_block?(matrix, blocks)
      blocks.any? do |block|
        next false unless block.size == 2
        first, last = block
        discriminant = complex_discriminant(matrix[first][first], matrix[first][last],
                                            matrix[last][first], matrix[last][last])
        next false unless discriminant < 0.0
        angle = Math.atan2(Math.sqrt(-discriminant),
                           matrix[first][first] + matrix[last][last]).abs
        angle > 2.5
      end
    end

    def schur_square_root(matrix)
      source = matrix.to_a
      blocks = root_schur_blocks(source)
      root = Array.new(@size) { Array.new(@size, 0.0) }
      blocks.each {|block| block_root!(root, source, block.first, block.last) }
      (1...blocks.size).each do |gap|
        (0...(blocks.size - gap)).each do |upper_index|
          solve_root_sylvester!(root, source, blocks, upper_index, upper_index + gap)
        end
      end
      @matrix.send(:new_matrix, root)
    end

    def root_schur_blocks(matrix)
      blocks = []
      tolerance = 100.0 * @epsilon * array_norm1(matrix)
      i = 0
      while i < @size
        if i + 1 < @size && matrix[i + 1][i].abs > tolerance &&
           complex_discriminant(matrix[i][i], matrix[i][i + 1],
                                matrix[i + 1][i], matrix[i + 1][i + 1]) < 0.0
          blocks << [i, i + 1]
          i += 2
        else
          matrix[i][i - 1] = 0.0 if i > 0 && !blocks.last&.include?(i - 1)
          blocks << [i, i]
          i += 1
        end
      end
      blocks
    end

    def block_root!(root, matrix, first, last)
      if first == last
        value = matrix[first][first]
        unless value > 0.0
          raise ArgumentError, "principal matrix logarithm is not defined for a non-positive real Schur block"
        end
        root[first][first] = Math.sqrt(value)
        return
      end

      a = matrix[first][first]
      b = matrix[first][last]
      c = matrix[last][first]
      d = matrix[last][last]
      determinant = a * d - b * c
      discriminant = complex_discriminant(a, b, c, d)

      if discriminant < 0.0
        gamma = Math.sqrt(determinant)
        delta = Math.sqrt(2.0 * gamma + a + d)
        alpha = gamma / delta
        beta = 1.0 / delta
      else
        trace_square_root = Math.sqrt(a + d + 2.0 * Math.sqrt([determinant, 0.0].max))
        if trace_square_root <= 0.0
          raise ArgumentError, "principal matrix logarithm is not defined for a non-positive real Schur block"
        end
        alpha = Math.sqrt([determinant, 0.0].max) / trace_square_root
        beta = 1.0 / trace_square_root
      end

      root[first][first] = alpha + beta * a
      root[first][last] = beta * b
      root[last][first] = beta * c
      root[last][last] = alpha + beta * d
    end

    def solve_root_sylvester!(root, matrix, blocks, upper_index, lower_index)
      upper = blocks[upper_index]
      lower = blocks[lower_index]
      left_size = block_size(upper)
      right_size = block_size(lower)
      unknown_count = left_size * right_size
      system = Array.new(unknown_count) { Array.new(unknown_count + 1, 0.0) }

      left_size.times do |row|
        right_size.times do |column|
          equation = row * right_size + column
          intermediate_sum = 0.0
          blocks[(upper_index + 1)...lower_index].each do |middle|
            block_size(middle).times do |entry|
              intermediate_sum += root[upper.first + row][middle.first + entry] *
                                   root[middle.first + entry][lower.first + column]
            end
          end
          left_size.times do |entry|
            system[equation][entry * right_size + column] +=
              root[upper.first + row][upper.first + entry]
          end
          right_size.times do |entry|
            system[equation][row * right_size + entry] +=
              root[lower.first + entry][lower.first + column]
          end
          system[equation][unknown_count] =
            matrix[upper.first + row][lower.first + column] - intermediate_sum
        end
      end

      values = gaussian_solve(system)
      left_size.times do |row|
        right_size.times do |column|
          root[upper.first + row][lower.first + column] =
            values[row * right_size + column]
        end
      end
    end

    def schur_logarithm(matrix, blocks)
      source = matrix.to_a
      result = Array.new(@size) { Array.new(@size, 0.0) }
      blocks.each {|block| block_logarithm!(result, source, block.first, block.last) }
      (1...blocks.size).each do |gap|
        (0...(blocks.size - gap)).each do |upper_index|
          solve_log_sylvester!(result, source, blocks, upper_index, upper_index + gap)
        end
      end
      @matrix.send(:new_matrix, result)
    end

    def schur_blocks(matrix)
      blocks = []
      tolerance = 100.0 * @epsilon * array_norm1(matrix)
      i = 0
      while i < @size
        if i + 1 < @size && matrix[i + 1][i].abs > tolerance
          blocks << [i, i + 1]
          i += 2
        else
          value = matrix[i][i]
          last = i
          while last + 1 < @size &&
                matrix[last + 1][last].abs <= tolerance &&
                (matrix[last + 1][last + 1] - value).abs <=
                  tolerance * [1.0, value.abs, matrix[last + 1][last + 1].abs].max
            last += 1
          end
          blocks << [i, last]
          i = last + 1
        end
      end
      blocks
    end

    def clear_real_splits(matrix)
      i = 0
      while i + 1 < @size
        discriminant = complex_discriminant(matrix[i][i], matrix[i][i + 1],
                                            matrix[i + 1][i], matrix[i + 1][i + 1])
        complex_tolerance = 10.0 * @epsilon * array_norm1(matrix) ** 2
        if matrix[i + 1][i].abs > 100.0 * @epsilon * array_norm1(matrix) &&
           discriminant > complex_tolerance
          matrix[i + 1][i] = 0.0
        end
        i += discriminant < 0.0 ? 2 : 1
      end
    end

    def block_logarithm!(result, matrix, first, last)
      if first == last
        value = matrix[first][first]
        unless value > 0.0
          raise ArgumentError, "principal matrix logarithm is not defined for a non-positive real eigenvalue"
        end
        result[first][first] = Math.log(value)
        return
      end

      a = matrix[first][first]
      b = matrix[first][last]
      c = matrix[last][first]
      d = matrix[last][last]
      discriminant = complex_discriminant(a, b, c, d)

      complex_tolerance = 10.0 * @epsilon * array_norm1(matrix) ** 2
      if first.succ == last && discriminant > complex_tolerance &&
         (a - d).abs > tolerance_value(matrix)
        coefficient = (Math.log(a) - Math.log(d)) / (a - d)
        constant = Math.log(a) - coefficient * a
        result[first][first] = constant + coefficient * a
        result[first][last] = coefficient * b
        result[last][first] = coefficient * c
        result[last][last] = constant + coefficient * d
      else
        if discriminant <= complex_tolerance
          determinant = a * d - b * c
          imaginary = Math.sqrt(-discriminant) / 2.0
          radius = Math.sqrt(determinant)
          real_log = 0.5 * Math.log(determinant)
          angle = Math.atan2(2.0 * imaginary, a + d)
          beta = angle / (radius * Math.sin(angle))
          result[first][first] = real_log
          result[last][last] = real_log
          result[first][last] = beta * b
          result[last][first] = beta * c
        else
          real_cluster_logarithm!(result, matrix, first, last)
        end
      end
    end

    def complex_discriminant(a, b, c, d)
      4.0 * b * c - (a - d) ** 2
    end

    def tolerance_value(matrix)
      100.0 * @epsilon * array_norm1(matrix)
    end

    def real_cluster_logarithm!(result, matrix, first, last)
      size = last - first + 1
      value = matrix[first][first]
      raise ArgumentError, "principal matrix logarithm is not defined for a non-positive real eigenvalue" unless value > 0.0

      nilpotent = Array.new(size) do |row|
        Array.new(size) do |column|
          entry = matrix[first + row][first + column]
          row == column ? entry - value : entry
        end
      end
      term = Array.new(size) {|row| Array.new(size) {|column| row == column ? 1.0 : 0.0 } }
      series = Array.new(size) { Array.new(size, 0.0) }
      size.times do |index|
        term = multiply_arrays(term, nilpotent)
        sign = index.even? ? 1.0 : -1.0
        series = add_arrays(series, multiply_array(term, sign / ((index + 1) * value ** (index + 1))))
      end
      size.times do |row|
        size.times do |column|
          result[first + row][first + column] =
            series[row][column] + (row == column ? Math.log(value) : 0.0)
        end
      end
    end

    def solve_log_sylvester!(result, matrix, blocks, upper_index, lower_index)
      upper = blocks[upper_index]
      lower = blocks[lower_index]
      left_size = block_size(upper)
      right_size = block_size(lower)
      unknown_count = left_size * right_size
      system = Array.new(unknown_count) { Array.new(unknown_count + 1, 0.0) }

      left_size.times do |row|
        right_size.times do |column|
          equation = row * right_size + column
          intermediate_sum = 0.0
          blocks[(upper_index + 1)...lower_index].each do |middle|
            block_size(middle).times do |entry|
              intermediate_sum += result[upper.first + row][middle.first + entry] *
                                   matrix[middle.first + entry][lower.first + column] +
                                   matrix[upper.first + row][middle.first + entry] *
                                   result[middle.first + entry][lower.first + column]
            end
          end
          left_size.times do |entry|
            system[equation][entry * right_size + column] +=
              result[upper.first + row][upper.first + entry]
          end
          right_size.times do |entry|
            system[equation][row * right_size + entry] +=
              result[lower.first + entry][lower.first + column]
          end
          system[equation][unknown_count] =
            matrix[upper.first + row][lower.first + column] - intermediate_sum
        end
      end

      values = gaussian_solve(system)
      left_size.times do |row|
        right_size.times do |column|
          result[upper.first + row][lower.first + column] =
            values[row * right_size + column]
        end
      end
    end

    def block_size(block)
      block.last - block.first + 1
    end

    def gaussian_solve(system)
      n = system.size
      n.times do |pivot_column|
        pivot_row = (pivot_column...n).max_by {|row| system[row][pivot_column].abs }
        pivot = system[pivot_row][pivot_column]
        if pivot.abs <= @epsilon * @epsilon
          raise RuntimeError, "unable to solve a Schur square-root Sylvester equation"
        end
        system[pivot_row], system[pivot_column] = system[pivot_column], system[pivot_row] if pivot_row != pivot_column
        n.times do |row|
          next if row == pivot_column
          factor = system[row][pivot_column] / pivot
          (pivot_column..n).each do |column|
            system[row][column] -= factor * system[pivot_column][column]
          end
        end
      end
      n.times.map {|row| system[row][n] / system[row][row] }
    end

    def array_norm1(matrix)
      (0...matrix.first.size).map do |column|
        matrix.sum {|row| row[column].abs }
      end.max.to_f
    end

    def subtract_identity(matrix)
      matrix.each_with_index.map do |row, row_index|
        row.each_with_index.map do |value, column_index|
          row_index == column_index ? value - 1.0 : value
        end
      end
    end

    def logarithm_series(matrix)
      term = Matrix.identity(@size)
      result = Matrix.zero(@size)
      SERIES_TERMS.times do |index|
        term = term * matrix
        coefficient = 1.0 / (index + 1)
        coefficient = -coefficient if index.odd?
        result = result + term * coefficient
      end
      result
    end

    def reconstruct_exponential(matrix, square_roots)
      term = Matrix.identity(@size)
      result = Matrix.identity(@size)
      EXPONENTIAL_TERMS.times do |index|
        term = term * matrix
        result = result + term / factorial(index + 1)
      end
      square_roots.times { result = result * result }
      result
    end

    def matrix_exponential(matrix)
      square_roots = [0, Math.log2(matrix_norm1(matrix) / 0.1).ceil].max
      scaled = matrix / (2.0 ** square_roots)
      term = Matrix.identity(@size)
      result = Matrix.identity(@size)
      EXPONENTIAL_TERMS.times do |index|
        term = term * scaled
        result = result + term / factorial(index + 1)
      end
      square_roots.times { result = result * result }
      result
    end

    def multiply_arrays(a, b)
      size = a.size
      Array.new(size) do |row|
        Array.new(size) do |column|
          (0...size).inject(0.0) {|sum, entry| sum + a[row][entry] * b[entry][column] }
        end
      end
    end

    def add_arrays(a, b)
      a.each_with_index.map do |row, row_index|
        row.each_with_index.map {|value, column| value + b[row_index][column] }
      end
    end

    def multiply_array(matrix, factor)
      matrix.map {|row| row.map {|value| value * factor } }
    end

    def factorial(value)
      (1..value).inject(1, :*)
    end

    def residual(matrix, schur_vectors, reconstructed_schur, log_schur, square_roots)
      reconstructed = schur_vectors * reconstructed_schur * schur_vectors.transpose
      log_matrix = schur_vectors * log_schur * schur_vectors.transpose
      residual_norm = matrix_norm1(reconstructed - matrix)
      relative_residual = residual_norm / [matrix_norm1(@matrix), Float::EPSILON].max
      {
        matrix: @matrix.send(:new_matrix, log_matrix.to_a),
        residual: residual_norm,
        relative_residual: relative_residual,
        square_roots: square_roots,
        series_terms: SERIES_TERMS
      }
    end

    def final_result(data)
      @diagnostics ? data : data.fetch(:matrix)
    end

    def failed_result(data)
      return data if @diagnostics
      raise RuntimeError,
        "could not reach requested logarithm residual precision #{@precision}; got #{data[:relative_residual]}"
    end

    def matrix_norm1(matrix)
      matrix.column_vectors.map {|column| column.to_a.sum(&:abs) }.max.to_f
    end
  end
end
